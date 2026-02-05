'use client';

import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/contexts/AuthContext';

/**
 * Price Sources with strict environment constraints
 */
export type PriceSource = 'LIVE_EXCHANGE_TICKER' | 'TESTNET_TICKER' | 'DERIVED_PRICE' | 'REPLAY_PRICE' | 'UNKNOWN';

export interface PriceData {
    symbol: string;
    price: number;
    source: PriceSource;
    exchange: string;
    environment: string;
    timestamp_ms: number;
    exchange_account_id?: string;
    bid?: number;
    ask?: number;
    age_ms?: number;
    is_stale?: boolean;
    is_valid?: boolean;
    rejection_reason?: string;
}

// Max age in milliseconds before price is considered stale
const MAX_AGE_MS: Record<PriceSource, number> = {
    LIVE_EXCHANGE_TICKER: 2000,    // 2 seconds
    TESTNET_TICKER: 5000,           // 5 seconds
    DERIVED_PRICE: 10000,           // 10 seconds
    REPLAY_PRICE: 60000,            // 1 minute
    UNKNOWN: 0,                     // Always stale
};

// Environment restrictions
const SOURCE_ENVIRONMENT_ALLOWED: Record<PriceSource, string[]> = {
    LIVE_EXCHANGE_TICKER: ['LIVE', 'live'],
    TESTNET_TICKER: ['TESTNET', 'testnet'],
    DERIVED_PRICE: ['LIVE', 'live', 'TESTNET', 'testnet'],
    REPLAY_PRICE: ['REPLAY', 'replay'],
    UNKNOWN: [],
};


/**
 * Calculate price age in milliseconds
 */
export function getPriceAge(timestamp_ms: number): number {
    return Date.now() - timestamp_ms;
}


/**
 * Check if price is stale based on source
 */
export function isStale(price: PriceData): boolean {
    const maxAge = MAX_AGE_MS[price.source] || 0;
    return getPriceAge(price.timestamp_ms) > maxAge;
}


/**
 * Validate price against environment
 */
export function validatePriceEnvironment(price: PriceData, expectedEnv: string): boolean {
    const allowed = SOURCE_ENVIRONMENT_ALLOWED[price.source] || [];
    return allowed.includes(expectedEnv) || allowed.includes(expectedEnv.toUpperCase());
}


/**
 * Price validation hook
 * 
 * HARD GUARDS:
 * - Drop price if source != expected
 * - Drop price if environment mismatch
 * - Drop price if timestamp stale
 * - Drop price if account mismatch
 */
export function usePriceValidation() {
    const { activeAccount } = useAuth();

    const validatePrice = useCallback((price: PriceData): PriceData | null => {
        if (!activeAccount) {
            return null;
        }

        // Guard 1: Check source is known
        if (price.source === 'UNKNOWN') {
            console.warn(`⚠️ Price dropped: unknown source for ${price.symbol}`);
            return null;
        }

        // Guard 2: Check environment match
        if (!validatePriceEnvironment(price, activeAccount.environment)) {
            console.warn(`⚠️ Price dropped: ${price.source} not allowed in ${activeAccount.environment}`);
            return null;
        }

        // Guard 3: Check staleness
        if (isStale(price)) {
            console.warn(`⚠️ Price dropped: stale (${getPriceAge(price.timestamp_ms)}ms old)`);
            return null;
        }

        // Guard 4: Check account mismatch
        if (price.exchange_account_id && price.exchange_account_id !== activeAccount.exchange_account_id) {
            console.warn(`⚠️ Price dropped: account mismatch`);
            return null;
        }

        return {
            ...price,
            age_ms: getPriceAge(price.timestamp_ms),
            is_stale: false,
            is_valid: true
        };
    }, [activeAccount]);

    return {
        validatePrice,
        activeEnvironment: activeAccount?.environment,
        activeAccountId: activeAccount?.exchange_account_id
    };
}


/**
 * Account-scoped price store
 * 
 * MANDATORY ACTIONS on account switch:
 * 1. Clear price store
 * 2. Block rendering until first valid tick
 */
export function usePriceStore() {
    const { activeAccount } = useAuth();
    const { validatePrice } = usePriceValidation();
    const [prices, setPrices] = useState<Record<string, PriceData>>({});
    const [hasFirstTick, setHasFirstTick] = useState(false);

    // Clear prices on account switch
    useEffect(() => {
        console.log('🔄 Price store cleared for new account');
        setPrices({});
        setHasFirstTick(false);
    }, [activeAccount?.exchange_account_id]);

    // Listen for accountSwitched event
    useEffect(() => {
        const handleAccountSwitch = () => {
            console.log('🔄 Account switched - clearing price store');
            setPrices({});
            setHasFirstTick(false);
        };

        window.addEventListener('accountSwitched', handleAccountSwitch);
        return () => window.removeEventListener('accountSwitched', handleAccountSwitch);
    }, []);

    const updatePrice = useCallback((price: PriceData) => {
        const validated = validatePrice(price);
        if (!validated) return false;

        setPrices(prev => ({
            ...prev,
            [validated.symbol]: validated
        }));
        setHasFirstTick(true);
        return true;
    }, [validatePrice]);

    const getPrice = useCallback((symbol: string): PriceData | null => {
        const price = prices[symbol.toUpperCase()];
        if (!price) return null;

        // Re-validate staleness
        if (isStale(price)) {
            return {
                ...price,
                is_stale: true,
                is_valid: false,
                rejection_reason: `Price stale: ${getPriceAge(price.timestamp_ms)}ms old`
            };
        }

        return price;
    }, [prices]);

    return {
        prices,
        hasFirstTick,
        updatePrice,
        getPrice,
        clearPrices: () => {
            setPrices({});
            setHasFirstTick(false);
        }
    };
}

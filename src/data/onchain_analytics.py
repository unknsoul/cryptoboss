"""
On-Chain Analytics Integration
Track whale movements, exchange flows, and on-chain metrics for trading signals.

Data Sources:
- Glassnode API (premium on-chain data)
- CryptoQuant API (exchange flows)
- Blockchain.com API (free blockchain data)
- Etherscan API (Ethereum on-chain)

Metrics:
- Exchange inflow/outflow (sell/buy pressure)
- Whale wallet movements (>1000 BTC)
- UTXO age (HODLing behavior)
- Miner behavior (selling/accumulating)
- Stablecoin supply (market sentiment)
"""

import requests
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class OnChainAnalytics:
    """
    On-chain analytics for crypto trading signals.
    
    Features:
    - Whale wallet tracking
    - Exchange flow monitoring
    - UTXO analysis
    - Miner behavior
    - Stablecoin metrics
    """
    
    def __init__(
        self,
        glassnode_api_key: Optional[str] = None,
        cryptoquant_api_key: Optional[str] = None,
        etherscan_api_key: Optional[str] = None
    ):
        """
        Initialize on-chain analytics.
        
        Args:
            glassnode_api_key: Glassnode API key (premium data)
            cryptoquant_api_key: CryptoQuant API key
            etherscan_api_key: Etherscan API key
        """
        self.glassnode_key = glassnode_api_key
        self.cryptoquant_key = cryptoquant_api_key
        self.etherscan_key = etherscan_api_key
        
        # Whale tracking (addresses with >1000 BTC)
        self.whale_addresses: List[str] = []
        self.whale_movements: List[Dict] = []
        
        logger.info("On-chain analytics initialized")
    
    def get_exchange_flows(self, symbol: str = "BTC", days: int = 7) -> Dict:
        """
        Get exchange inflow/outflow data.
        
        Interpretation:
        - High inflow = Selling pressure (bearish)
        - High outflow = Accumulation (bullish)
        
        Returns:
            Dict with inflow, outflow, net_flow
        """
        try:
            # Using CryptoQuant API
            if self.cryptoquant_key:
                url = f"https://api.cryptoquant.com/v1/btc/exchange-flows/inflow-total"
                headers = {"Authorization": f"Bearer {self.cryptoquant_key}"}
                params = {"window": "day", "limit": days}
                
                response = requests.get(url, headers=headers, params=params)
                inflow_data = response.json()
                
                url = f"https://api.cryptoquant.com/v1/btc/exchange-flows/outflow-total"
                response = requests.get(url, headers=headers, params=params)
                outflow_data = response.json()
                
                # Calculate metrics
                total_inflow = sum([d['value'] for d in inflow_data['result']['data']])
                total_outflow = sum([d['value'] for d in outflow_data['result']['data']])
                net_flow = total_inflow - total_outflow
                
                # Signal interpretation
                signal = "NEUTRAL"
                if net_flow > 10000:  # More than 10k BTC into exchanges
                    signal = "BEARISH"
                elif net_flow < -10000:  # More than 10k BTC out of exchanges
                    signal = "BULLISH"
                
                return {
                    'inflow': total_inflow,
                    'outflow': total_outflow,
                    'net_flow': net_flow,
                    'signal': signal,
                    'days': days
                }
            else:
                # Fallback: Use blockchain.com API (free)
                url = "https://api.blockchain.info/charts/n-transactions"
                params = {"timespan": f"{days}days", "format": "json"}
                response = requests.get(url, params=params)
                data = response.json()
                
                # Estimate flows from transaction volume
                avg_tx = sum([d['y'] for d in data['values']]) / len(data['values'])
                
                return {
                    'avg_transactions_per_day': avg_tx,
                    'signal': 'NEUTRAL',
                    'note': 'Limited data without API key'
                }
                
        except Exception as e:
            logger.error(f"Error fetching exchange flows: {e}")
            return {'error': str(e)}
    
    def track_whale_movements(self, min_btc: float = 1000) -> List[Dict]:
        """
        Track large BTC transfers (whale movements).
        
        Args:
            min_btc: Minimum BTC for whale transaction
        
        Returns:
            List of whale transactions
        """
        try:
            # Using blockchain.com API
            url = "https://blockchain.info/unconfirmed-transactions?format=json"
            response = requests.get(url)
            data = response.json()
            
            whale_txs = []
            
            for tx in data.get('txs', []):
                # Calculate total BTC moved
                total_btc = sum([out['value'] for out in tx.get('out', [])]) / 1e8
                
                if total_btc >= min_btc:
                    whale_txs.append({
                        'hash': tx['hash'],
                        'amount_btc': total_btc,
                        'time': datetime.fromtimestamp(tx['time']),
                        'from_address': tx.get('inputs', [{}])[0].get('prev_out', {}).get('addr'),
                        'to_addresses': [out.get('addr') for out in tx.get('out', [])]
                    })
            
            self.whale_movements.extend(whale_txs)
            
            logger.info(f"Found {len(whale_txs)} whale transactions")
            
            return whale_txs
            
        except Exception as e:
            logger.error(f"Error tracking whales: {e}")
            return []
    
    def get_utxo_age_distribution(self) -> Dict:
        """
        Get UTXO age distribution (HODLing analysis).
        
        Interpretation:
        - Increasing old coins = Strong hands holding (bullish)
        - Decreasing old coins = Old hands selling (bearish)
        
        Returns:
            Dict with age distribution
        """
        try:
            if self.glassnode_key:
                url = "https://api.glassnode.com/v1/metrics/supply/hodl_waves"
                params = {
                    'a': 'BTC',
                    'api_key': self.glassnode_key,
                    'i': '24h'
                }
                
                response = requests.get(url, params=params)
                data = response.json()
                
                # Latest data point
                latest = data[-1] if data else {}
                
                return {
                    'timestamp': latest.get('t'),
                    'hodl_waves': latest.get('o', {}),
                    'interpretation': self._interpret_hodl_waves(latest.get('o', {}))
                }
            else:
                return {'error': 'Glassnode API key required'}
                
        except Exception as e:
            logger.error(f"Error fetching UTXO age: {e}")
            return {'error': str(e)}
    
    def _interpret_hodl_waves(self, waves: Dict) -> str:
        """Interpret HODL waves for signal."""
        # HODL waves show % of supply by age
        # If old coins (>1year) increasing = bullish
        old_coins_pct = sum([v for k, v in waves.items() if '1y' in k or '2y' in k])
        
        if old_coins_pct > 65:
            return "STRONG_HODL (Bullish)"
        elif old_coins_pct > 50:
            return "MODERATE_HODL (Neutral-Bullish)"
        else:
            return "WEAK_HODL (Neutral-Bearish)"
    
    def get_miner_behavior(self) -> Dict:
        """
        Get miner sell/accumulate behavior.
        
        Interpretation:
        - Miners selling = Bearish pressure
        - Miners accumulating = Confident/bullish
        
        Returns:
            Dict with miner metrics
        """
        try:
            if self.glassnode_key:
                # Miner net position change
                url = "https://api.glassnode.com/v1/metrics/mining/revenue_from_fees"
                params = {
                    'a': 'BTC',
                    'api_key': self.glassnode_key,
                    'i': '24h'
                }
                
                response = requests.get(url, params=params)
                data = response.json()
                
                recent_revenue = data[-7:] if len(data) >= 7 else data
                avg_revenue = sum([d['v'] for d in recent_revenue]) / len(recent_revenue)
                
                return {
                    'avg_daily_revenue_btc': avg_revenue,
                    'interpretation': 'Miners earning well' if avg_revenue > 50 else 'Low miner revenue'
                }
            else:
                return {'error': 'Glassnode API key required'}
                
        except Exception as e:
            logger.error(f"Error fetching miner data: {e}")
            return {'error': str(e)}
    
    def get_stablecoin_supply(self) -> Dict:
        """
        Get stablecoin supply metrics.
        
        Interpretation:
        - Increasing supply = More dry powder to buy crypto (bullish)
        - Decreasing supply = capital leaving market (bearish)
        
        Returns:
            Dict with stablecoin metrics
        """
        try:
            # Using CoinGecko API (free)
            stablecoins = ['tether', 'usd-coin', 'binance-usd', 'dai']
            total_mcap = 0
            
            for coin in stablecoins:
                url = f"https://api.coingecko.com/api/v3/coins/{coin}"
                response = requests.get(url)
                data = response.json()
                
                mcap = data.get('market_data', {}).get('market_cap', {}).get('usd', 0)
                total_mcap += mcap
            
            # Get historical for comparison
            url = "https://api.coingecko.com/api/v3/coins/tether/market_chart"
            params = {'vs_currency': 'usd', 'days': 30}
            response = requests.get(url, params=params)
            hist_data = response.json()
            
            mcap_30d_ago = hist_data['market_caps'][0][1]
            current_mcap = hist_data['market_caps'][-1][1]
            change_pct = ((current_mcap - mcap_30d_ago) / mcap_30d_ago) * 100
            
            signal = "NEUTRAL"
            if change_pct > 5:
                signal = "BULLISH"
            elif change_pct < -5:
                signal = "BEARISH"
            
            return {
                'total_stablecoin_mcap': total_mcap,
                'mcap_change_30d_pct': change_pct,
                'signal': signal,
                'interpretation': f"Stablecoin supply {change_pct:+.1f}% in 30 days"
            }
            
        except Exception as e:
            logger.error(f"Error fetching stablecoin data: {e}")
            return {'error': str(e)}
    
    def get_comprehensive_signal(self) -> Dict:
        """
        Get comprehensive on-chain trading signal.
        
        Combines all metrics for overall signal.
        
        Returns:
            Dict with overall signal and component scores
        """
        signals = {}
        
        # Get all metrics
        exchange_flow = self.get_exchange_flows()
        exchange_signal = exchange_flow.get('signal', 'NEUTRAL')
        signals['exchange_flow'] = exchange_signal
        
        utxo = self.get_utxo_age_distribution()
        if 'STRONG_HODL' in utxo.get('interpretation', ''):
            signals['utxo'] = 'BULLISH'
        elif 'WEAK_HODL' in utxo.get('interpretation', ''):
            signals['utxo'] = 'BEARISH'
        else:
            signals['utxo'] = 'NEUTRAL'
        
        stablecoin = self.get_stablecoin_supply()
        signals['stablecoin'] = stablecoin.get('signal', 'NEUTRAL')
        
        # Calculate overall signal
        bullish_count = sum(1 for s in signals.values() if s == 'BULLISH')
        bearish_count = sum(1 for s in signals.values() if s == 'BEARISH')
        
        if bullish_count >= 2:
            overall = 'BULLISH'
            confidence = bullish_count / len(signals)
        elif bearish_count >= 2:
            overall = 'BEARISH'
            confidence = bearish_count / len(signals)
        else:
            overall = 'NEUTRAL'
            confidence = 0.5
        
        return {
            'overall_signal': overall,
            'confidence': confidence,
            'component_signals': signals,
            'timestamp': datetime.now().isoformat()
        }


# Example usage
if __name__ == "__main__":
    # Initialize
    analytics = OnChainAnalytics(
        glassnode_api_key="YOUR_GLASSNODE_KEY",  # Optional
        cryptoquant_api_key="YOUR_CRYPTOQUANT_KEY"  # Optional
    )
    
    # Get comprehensive signal
    signal = analytics.get_comprehensive_signal()
    print(f"On-Chain Signal: {signal['overall_signal']} ({signal['confidence']:.0%} confidence)")
    print(f"Components: {signal['component_signals']}")
    
    # Track whales
    whales = analytics.track_whale_movements(min_btc=1000)
    print(f"\nWhale Movements: {len(whales)} large transactions")
    
    # Exchange flows
    flows = analytics.get_exchange_flows(days=7)
    print(f"\nExchange Flow Signal: {flows.get('signal')}")
    print(f"Net Flow: {flows.get('net_flow', 0):,.0f} BTC")

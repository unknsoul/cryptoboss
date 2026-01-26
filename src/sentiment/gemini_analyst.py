"""
Gemini News Analyst
Uses Google's Gemini Pro to analyze crypto news sentiment and generate fundamental signals.
"""

import os
import random
from typing import List, Dict, Optional, Any
import google.generativeai as genai
from datetime import datetime

class GeminiNewsAnalyst:
    def __init__(self, api_key: str = None):
        """
        Initialize Gemini Analyst.
        Auto-detects GOOGLE_API_KEY from environment if not passed.
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.model = None
        self.is_active = False
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                self.is_active = True
                print("✅ Gemini AI Connected")
            except Exception as e:
                print(f"⚠️ Gemini Connection Failed: {e}")
        else:
            print("⚠️ No Gemini API Key found. Using Mock Mode.")

    def fetch_news(self, symbol: str = "BTC") -> List[str]:
        """
        Fetch news headlines (Mock implementation for stability).
        In production, connect to CryptoCompare/NewsAPI.
        """
        # Simulated news feed for demonstration
        bullish_news = [
            f"{symbol} ETF inflows hit record highs",
            f"Major bank announces {symbol} custody services",
            f"{symbol} hashrate hits all-time high",
            "Regulatory clarity improves in key markets",
            f"Whales accumulating {symbol} at current levels"
        ]
        
        bearish_news = [
            f"SEC delays {symbol} ETF decision",
            f"Exchange hack results in {symbol} theft",
            "Macro fears grip markets, risk-off sentiment prevails",
            f"Miner capitulation fears for {symbol}",
            "Technicals weaken as support breaks"
        ]
        
        neutral_news = [
            f"{symbol} trades sideways amidst low volatility",
            "Market awaits Fed meeting minutes",
            f"Development activity on {symbol} network stable",
            "Analyst predicts price movement soon",
            f"{symbol} dominance remains steady"
        ]
        
        # Randomly mix news based on a "hidden" market cycle (random for now)
        # Real impl would hit an API
        bias = random.random()
        if bias > 0.6:
            headlines = random.sample(bullish_news, 2) + random.sample(neutral_news, 1)
        elif bias < 0.4:
            headlines = random.sample(bearish_news, 2) + random.sample(neutral_news, 1)
        else:
            headlines = random.sample(neutral_news, 3)
            
        return headlines

    def analyze_sentiment(self, headlines: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment of headlines using Gemini.
        Returns: {score: float (-1 to 1), reasoning: str}
        """
        if not self.is_active or not self.model:
            return self._mock_analysis(headlines)
            
        prompt = f"""
        Analyze the sentiment of these crypto news headlines for a professional trading algorithm.
        Headlines:
        {chr(10).join(['- ' + h for h in headlines])}

        Provide a JSON response with:
        1. "score": A float from -1.0 (Very Bearish) to 1.0 (Very Bullish).
        2. "reasoning": A brief 1-sentence explanation.
        3. "signal": "BUY", "SELL", or "HOLD".
        
        Be conservative. Only give high scores for very significant news.
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Simple text parsing if JSON fails (Gemini can be chatty)
            # Ideally use response.text and json.loads(), but for robustness:
            text = response.text.lower()
            
            score = 0.0
            if "score" in text:
                # varied parsing logic would go here
                pass
                
            # Fallback to simple keyword scan of the LLM's own output if JSON parsing is complex to impl in one file
            # For this MVP, we will trust the LLM to output a number if asked, but let's use a robust fallback
            # Implementing a safer 'mock' override if the parsing is too fragile for a single file demo
            # But let's try to extract a number
            import re
            match = re.search(r"score\":\s*([-+]?\d*\.\d+|\d+)", response.text)
            if match:
                score = float(match.group(1))
                reasoning = "Gemini Analysis Successful"
                signal = "HOLD"
                if score > 0.5: signal = "BUY"
                if score < -0.5: signal = "SELL"
                
                return {
                    "score": max(min(score, 1.0), -1.0),
                    "reasoning": reasoning, 
                    "signal": signal,
                    "source": "Gemini-Pro"
                }
            else:
                 return self._mock_analysis(headlines)

        except Exception as e:
            print(f"Gemini Analysis Error: {e}")
            return self._mock_analysis(headlines)

    def _mock_analysis(self, headlines: List[str]) -> Dict[str, Any]:
        """Fallback method using simple keyword matching"""
        score = 0
        bull_words = ['record', 'high', 'inflow', 'accumulat', 'clarity', 'bull']
        bear_words = ['delay', 'hack', 'fear', 'capitulat', 'weak', 'bear']
        
        text = " ".join(headlines).lower()
        
        for w in bull_words: 
            if w in text: score += 0.3
        for w in bear_words: 
            if w in text: score -= 0.3
            
        score = max(min(score, 1.0), -1.0)
        
        signal = "HOLD"
        if score >= 0.3: signal = "BUY"
        elif score <= -0.3: signal = "SELL"
        
        return {
            "score": score, 
            "reasoning": "Keyword Analysis (Fallback)",
            "signal": signal,
            "source": "Mock-Engine"
        }

if __name__ == "__main__":
    analyst = GeminiNewsAnalyst()
    news = analyst.fetch_news("BTC")
    print("Headlines:", news)
    print("Analysis:", analyst.analyze_sentiment(news))

"""
Vanguard-Alpha Trading Brain v2.0
==================================
المحرك المركزي الذكي للنظام - Master Orchestrator

المكونات:
1. Market State Machine - آلة حالات السوق الديناميكية
2. Feature Engineering Layer - طبقة استخراج الميزات المتقدمة
3. Decision Orchestrator - منسق القرارات
4. Risk Coordinator - منسق المخاطر المتقدم
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass
from collections import deque

# =============== ENUMS & CONFIGURATIONS ===============

class MarketState(Enum):
    """حالات السوق الديناميكية"""
    NORMAL = "normal"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    HIGH_VOLATILITY = "high_volatility"
    RANGING = "ranging"
    CRISIS = "crisis"
    MAINTENANCE = "maintenance"
    EMERGENCY_HALT = "emergency_halt"

class TradingMode(Enum):
    """أوضاع التداول"""
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    HALT = "halt"

@dataclass
class SystemConfig:
    """إعدادات النظام الشاملة"""
    # Trading Parameters
    symbol: str = "BTCUSD"
    base_position_size: float = 0.001
    
    # AI Configuration
    ai_confidence_threshold: float = 0.70
    feature_dim: int = 20
    sequence_length: int = 50
    
    # Risk Limits
    max_position_size: float = 0.01
    max_daily_loss: float = 500.0
    max_drawdown: float = 0.15
    max_latency_ms: int = 500
    max_trades_per_minute: int = 10
    max_spread: float = 0.001
    
    # Market State Thresholds
    high_volatility_threshold: float = 0.03
    emergency_loss_threshold: float = 1000.0
    
    # Feature Engineering
    sma_periods: List[int] = None
    ema_periods: List[int] = None
    
    def __post_init__(self):
        if self.sma_periods is None:
            self.sma_periods = [5, 20, 50]
        if self.ema_periods is None:
            self.ema_periods = [12, 26]

# =============== FEATURE ENGINEERING LAYER ===============

class FeatureEngineer:
    """طبقة استخراج الميزات المتقدمة"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger("FeatureEngineer")
        
    def extract_features(self, market_data) -> Optional[Dict]:
        """استخراج جميع الميزات من بيانات السوق"""
        
        try:
            if len(market_data.prices) < max(self.config.sma_periods):
                return None
            
            prices = np.array(list(market_data.prices))
            volumes = np.array(list(market_data.volumes)) if hasattr(market_data, 'volumes') else None
            
            features = {}
            
            # 1. Price Features
            features['price'] = prices[-1]
            features['price_change_pct'] = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
            
            # 2. Moving Averages
            for period in self.config.sma_periods:
                if len(prices) >= period:
                    features[f'price_sma_{period}'] = np.mean(prices[-period:])
            
            # 3. Exponential Moving Averages
            for period in self.config.ema_periods:
                if len(prices) >= period:
                    features[f'price_ema_{period}'] = self._calculate_ema(prices, period)
            
            # 4. RSI
            features['rsi'] = self._calculate_rsi(prices, period=14)
            
            # 5. MACD
            if len(prices) >= 26:
                ema_12 = self._calculate_ema(prices, 12)
                ema_26 = self._calculate_ema(prices, 26)
                features['macd'] = ema_12 - ema_26
                features['macd_signal'] = self._calculate_ema(np.array([features['macd']]), 9)
                features['macd_hist'] = features['macd'] - features['macd_signal']
            
            # 6. Volatility
            if len(prices) >= 20:
                returns = np.diff(prices[-20:]) / prices[-21:-1]
                features['volatility'] = np.std(returns)
            else:
                features['volatility'] = 0.02
            
            # 7. Bollinger Bands
            if len(prices) >= 20:
                sma_20 = np.mean(prices[-20:])
                std_20 = np.std(prices[-20:])
                features['bb_upper'] = sma_20 + (2 * std_20)
                features['bb_lower'] = sma_20 - (2 * std_20)
                features['bb_width'] = features['bb_upper'] - features['bb_lower']
            
            # 8. Volume Features (if available)
            if volumes is not None and len(volumes) >= 20:
                features['volume'] = volumes[-1]
                features['volume_sma_20'] = np.mean(volumes[-20:])
                features['volume_ratio'] = volumes[-1] / features['volume_sma_20'] if features['volume_sma_20'] > 0 else 1.0
            
            # 9. Advanced Features (placeholders for future)
            features['spread'] = 0.0005  # سيتم حسابه من Order Book
            features['order_flow'] = 0.0  # سيتم حسابه من Tick Data
            features['market_depth'] = 1.0  # سيتم حسابه من Order Book
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return None
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """حساب مؤشر القوة النسبية"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """حساب المتوسط المتحرك الأسي"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema

# =============== MARKET STATE MACHINE ===============

class MarketStateMachine:
    """آلة حالات السوق الديناميكية"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.current_state = MarketState.NORMAL
        self.logger = logging.getLogger("MarketStateMachine")
        self.state_history = deque(maxlen=100)
    
    def analyze_market_conditions(self, features: Dict) -> MarketState:
        """تحليل ظروف السوق وتحديد الحالة"""
        
        if features is None:
            return MarketState.MAINTENANCE
        
        new_state = MarketState.NORMAL
        
        # فحص التقلب
        if features.get('volatility', 0) > self.config.high_volatility_threshold:
            self.logger.warning(f"⚠️ High volatility detected: {features['volatility']:.4f}")
            new_state = MarketState.HIGH_VOLATILITY
        
        # فحص الاتجاه الصاعد
        elif (features.get('price_sma_20', 0) > features.get('price_sma_50', 0) and 
              features.get('rsi', 50) > 55):
            new_state = MarketState.TRENDING_UP
        
        # فحص الاتجاه الهابط
        elif (features.get('price_sma_20', 0) < features.get('price_sma_50', 0) and 
              features.get('rsi', 50) < 45):
            new_state = MarketState.TRENDING_DOWN
        
        # فحص السوق العرضي
        elif abs(features.get('price_change_pct', 0)) < 0.005:
            new_state = MarketState.RANGING
        
        # تسجيل التغيير في الحالة
        if new_state != self.current_state:
            self.logger.info(f"🔄 Market state changed: {self.current_state.value} → {new_state.value}")
            self.current_state = new_state
            self.state_history.append({
                'timestamp': datetime.now(),
                'state': new_state,
                'features': features.copy()
            })
        
        return self.current_state
    
    def get_trading_mode(self, state: MarketState, risk_metrics: Dict) -> TradingMode:
        """تحديد وضع التداول بناءً على حالة السوق"""
        
        # فحص حالة الطوارئ
        if risk_metrics.get('daily_pnl', 0) < -self.config.emergency_loss_threshold:
            self.logger.critical("🚨 EMERGENCY HALT: Loss threshold exceeded!")
            return TradingMode.HALT
        
        # فحص الحالات الحرجة
        if state == MarketState.CRISIS or state == MarketState.EMERGENCY_HALT:
            return TradingMode.HALT
        
        elif state == MarketState.HIGH_VOLATILITY:
            return TradingMode.CONSERVATIVE
        
        elif state in [MarketState.TRENDING_UP, MarketState.TRENDING_DOWN]:
            return TradingMode.AGGRESSIVE
        
        else:
            return TradingMode.NORMAL

# =============== PRO RISK ENGINE ===============

class ProRiskEngine:
    """محرك المخاطر الاحترافي - 9 فحوصات حرجة"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger("ProRiskEngine")
        
        # Risk metrics
        self.daily_pnl = 0.0
        self.peak_equity = 10000.0
        self.current_equity = 10000.0
        self.trade_count_minute = 0
        self.last_minute_reset = datetime.now()
        
        # Position tracking
        self.open_positions = {}
        self.total_exposure = 0.0
        
        # Trade history
        self.trade_history = deque(maxlen=1000)
    
    def can_trade(self, 
                   signal: Dict,
                   features: Dict,
                   latency_ms: int,
                   trading_mode: TradingMode) -> Tuple[bool, str]:
        """فحص شامل لإمكانية التداول - 9 فحوصات"""
        
        # 1. فحص وضع التداول
        if trading_mode == TradingMode.HALT:
            return False, "Trading halted"
        
        # 2. فحص Latency
        if latency_ms > self.config.max_latency_ms:
            self.logger.warning(f"🚫 High latency: {latency_ms}ms")
            return False, f"Latency too high: {latency_ms}ms"
        
        # 3. فحص الخسارة اليومية
        if self.daily_pnl < -self.config.max_daily_loss:
            self.logger.error(f"🛑 Max daily loss reached: ${self.daily_pnl:.2f}")
            return False, "Max daily loss exceeded"
        
        # 4. فحص Max Drawdown
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        if drawdown > self.config.max_drawdown:
            self.logger.error(f"🛑 Max drawdown: {drawdown:.2%}")
            return False, "Max drawdown exceeded"
        
        # 5. فحص معدل التداول
        if self._check_rate_limit():
            return False, "Rate limit exceeded"
        
        # 6. فحص حجم المركز
        if self.total_exposure >= self.config.max_position_size:
            return False, "Max position size reached"
        
        # 7. فحص Spread
        if features and features.get('spread', 0) > self.config.max_spread:
            return False, f"Spread too wide: {features['spread']:.4f}"
        
        # 8. فحص ثقة AI
        if signal.get('confidence', 0) < self.config.ai_confidence_threshold:
            return False, f"Low AI confidence: {signal.get('confidence', 0):.2%}"
        
        # 9. Circuit Breaker (فحص نهائي)
        if self._check_circuit_breaker():
            return False, "Circuit breaker activated"
        
        return True, "OK"
    
    def _check_rate_limit(self) -> bool:
        """فحص حد معدل التداول"""
        now = datetime.now()
        if (now - self.last_minute_reset).seconds >= 60:
            self.trade_count_minute = 0
            self.last_minute_reset = now
        
        if self.trade_count_minute >= self.config.max_trades_per_minute:
            return True
        
        return False
    
    def _check_circuit_breaker(self) -> bool:
        """فحص قاطع الدائرة"""
        # فحص إذا كان هناك خسائر متتالية كثيرة
        if len(self.trade_history) >= 5:
            recent_trades = list(self.trade_history)[-5:]
            losing_streak = all(t.get('pnl', 0) < 0 for t in recent_trades)
            if losing_streak:
                self.logger.critical("🚨 Circuit breaker: 5 consecutive losses!")
                return True
        
        return False
    
    def calculate_position_size(self, 
                                 signal: Dict,
                                 features: Dict,
                                 trading_mode: TradingMode) -> float:
        """حساب حجم الصفقة الأمثل"""
        
        base_size = self.config.base_position_size
        
        # تعديل بناءً على وضع التداول
        if trading_mode == TradingMode.CONSERVATIVE:
            size = base_size * 0.5
        elif trading_mode == TradingMode.AGGRESSIVE:
            size = base_size * 1.5
        else:
            size = base_size
        
        # تعديل بناءً على الثقة
        confidence = signal.get('confidence', 0.5)
        size *= confidence
        
        # تعديل بناءً على التقلب
        if features:
            volatility = features.get('volatility', 0.02)
            size *= (0.02 / max(volatility, 0.01))
        
        # التأكد من عدم تجاوز الحد الأقصى
        remaining_capacity = self.config.max_position_size - self.total_exposure
        size = min(size, remaining_capacity)
        
        return round(size, 4)
    
    def update_pnl(self, pnl: float):
        """تحديث الربح/الخسارة"""
        self.daily_pnl += pnl
        self.current_equity += pnl
        
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
    
    def record_trade(self, trade_data: Dict):
        """تسجيل الصفقة"""
        self.trade_history.append(trade_data)
        self.trade_count_minute += 1

# =============== TRADING BRAIN (MASTER ORCHESTRATOR) ===============

class TradingBrain:
    """العقل المركزي للنظام - Master Orchestrator"""
    
    def __init__(self, 
                 market_data,
                 ai_engine=None,
                 execution_service=None,
                 database_manager=None):
        
        self.config = SystemConfig()
        self.market_data = market_data
        self.ai_engine = ai_engine
        self.execution_service = execution_service
        self.database_manager = database_manager
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(self.config)
        self.state_machine = MarketStateMachine(self.config)
        self.risk_engine = ProRiskEngine(self.config)
        
        self.logger = logging.getLogger("TradingBrain")
        self.logger.info("🧠 Trading Brain initialized")
    
    async def process_tick(self, tick_data: Dict) -> Optional[Dict]:
        """معالجة Tick واتخاذ القرار"""
        
        try:
            # 1. Extract Features
            features = self.feature_engineer.extract_features(self.market_data)
            if features is None:
                return None
            
            # 2. Analyze Market State
            market_state = self.state_machine.analyze_market_conditions(features)
            
            # 3. Determine Trading Mode
            risk_metrics = {
                'daily_pnl': self.risk_engine.daily_pnl,
                'current_equity': self.risk_engine.current_equity
            }
            trading_mode = self.state_machine.get_trading_mode(market_state, risk_metrics)
            
            # 4. Get AI Signal
            signal = self._get_ai_signal(features)
            if signal is None or signal['action'] == 'HOLD':
                return None
            
            # 5. Risk Check
            latency_ms = tick_data.get('latency', 0)
            can_trade, reason = self.risk_engine.can_trade(
                signal, features, latency_ms, trading_mode
            )
            
            if not can_trade:
                self.logger.info(f"🚫 Trade blocked: {reason}")
                return None
            
            # 6. Calculate Position Size
            position_size = self.risk_engine.calculate_position_size(
                signal, features, trading_mode
            )
            
            # 7. Create Order
            order = {
                'symbol': self.config.symbol,
                'action': signal['action'],
                'quantity': position_size,
                'price': features['price'],
                'confidence': signal['confidence'],
                'market_state': market_state.value,
                'trading_mode': trading_mode.value,
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"✅ Order created: {signal['action']} {position_size} @ ${features['price']:.2f}")
            
            # 8. Execute (if service available)
            if self.execution_service:
                await self.execution_service.execute_order(order)
            
            # 9. Record Trade
            self.risk_engine.record_trade(order)
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error processing tick: {e}")
            return None
    
    def _get_ai_signal(self, features: Dict) -> Optional[Dict]:
        """الحصول على إشارة من AI Engine"""
        
        if self.ai_engine is None:
            # Fallback: استراتيجية بسيطة
            rsi = features.get('rsi', 50)
            if rsi < 30:
                return {'action': 'BUY', 'confidence': 0.75}
            elif rsi > 70:
                return {'action': 'SELL', 'confidence': 0.75}
            else:
                return {'action': 'HOLD', 'confidence': 0.5}
        
        # استخدام AI Engine الحقيقي
        try:
            return self.ai_engine.predict(features)
        except Exception as e:
            self.logger.error(f"AI Engine error: {e}")
            return None
    
    def get_status(self) -> Dict:
        """الحصول على حالة النظام"""
        return {
            'market_state': self.state_machine.current_state.value,
            'daily_pnl': self.risk_engine.daily_pnl,
            'current_equity': self.risk_engine.current_equity,
            'total_exposure': self.risk_engine.total_exposure,
            'trade_count': len(self.risk_engine.trade_history),
            'peak_equity': self.risk_engine.peak_equity
        }

# =============== EXAMPLE USAGE ===============

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Mock market data
    class MockMarketData:
        def __init__(self):
            self.prices = deque(maxlen=200)
            self.volumes = deque(maxlen=200)
            
            # Generate some data
            for i in range(100):
                self.prices.append(42000 + np.random.randn() * 100)
                self.volumes.append(1000 + np.random.randn() * 100)
    
    market_data = MockMarketData()
    brain = TradingBrain(market_data)
    
    # Test
    tick = {'price': 42000, 'latency': 100}
    
    import asyncio
    result = asyncio.run(brain.process_tick(tick))
    
    if result:
        print(f"\n✅ Order: {result}")
    
    status = brain.get_status()
    print(f"\n📊 Status: {status}")

"""
Vanguard-Alpha Stress Testing & Real-time Monitoring v2.0
=========================================================
نظام اختبار الضغط والمراقبة الحية

المكونات:
1. Stress Tester - اختبار 5 سيناريوهات حرجة
2. Real-time Monitor - مراقبة حية للنظام
3. Alert System - نظام تنبيهات
4. Performance Tracker - تتبع الأداء
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import json

# =============== CONFIGURATION ===============

@dataclass
class MonitorConfig:
    """إعدادات المراقبة"""
    # Alert thresholds
    max_latency_warning: int = 300  # ms
    max_latency_critical: int = 500  # ms
    
    max_drawdown_warning: float = 0.10  # 10%
    max_drawdown_critical: float = 0.15  # 15%
    
    daily_loss_warning: float = 300.0  # $
    daily_loss_critical: float = 500.0  # $
    
    # Monitoring intervals
    monitor_interval_seconds: int = 5
    report_interval_seconds: int = 60
    
    # History size
    metrics_history_size: int = 1000

# =============== STRESS TESTER ===============

class StressTester:
    """اختبار الضغط - 5 سيناريوهات حرجة"""
    
    def __init__(self, trading_brain):
        self.trading_brain = trading_brain
        self.logger = logging.getLogger("StressTester")
        self.test_results = {}
    
    async def run_all_scenarios(self) -> Dict:
        """تشغيل جميع سيناريوهات الاختبار"""
        
        self.logger.info("🔬 Starting comprehensive stress testing...")
        
        scenarios = [
            self.test_high_volatility,
            self.test_flash_crash,
            self.test_high_latency,
            self.test_liquidity_crisis,
            self.test_api_failure
        ]
        
        for scenario in scenarios:
            try:
                result = await scenario()
                self.test_results[scenario.__name__] = result
            except Exception as e:
                self.logger.error(f"❌ {scenario.__name__} failed: {e}")
                self.test_results[scenario.__name__] = {'status': 'FAILED', 'error': str(e)}
        
        self.logger.info("✅ Stress testing complete")
        return self.test_results
    
    async def test_high_volatility(self) -> Dict:
        """اختبار 1: تقلب عالي"""
        
        self.logger.info("🌪️ Testing HIGH VOLATILITY scenario...")
        
        # محاكاة تقلب عالي (5% في دقيقة)
        test_data = {
            'price': 42000,
            'volatility': 0.05,  # 5% (عالي جداً)
            'rsi': 75,
            'latency': 100
        }
        
        # محاكاة features
        features = {
            'price': test_data['price'],
            'volatility': test_data['volatility'],
            'rsi': test_data['rsi'],
            'price_sma_20': 42000,
            'price_sma_50': 41800,
            'price_change_pct': 0.05
        }
        
        # تحليل حالة السوق
        market_state = self.trading_brain.state_machine.analyze_market_conditions(features)
        
        # التحقق من الاستجابة
        risk_metrics = {'daily_pnl': 0}
        trading_mode = self.trading_brain.state_machine.get_trading_mode(market_state, risk_metrics)
        
        result = {
            'status': 'PASSED',
            'market_state': market_state.value,
            'trading_mode': trading_mode.value,
            'expected_mode': 'conservative',
            'test_passed': trading_mode.value == 'conservative'
        }
        
        if result['test_passed']:
            self.logger.info("✅ High volatility test PASSED - System switched to conservative mode")
        else:
            self.logger.warning(f"⚠️ High volatility test WARNING - Expected conservative, got {trading_mode.value}")
        
        return result
    
    async def test_flash_crash(self) -> Dict:
        """اختبار 2: انهيار مفاجئ"""
        
        self.logger.info("💥 Testing FLASH CRASH scenario...")
        
        # محاكاة انهيار (-20% في ثواني)
        features = {
            'price': 33600,  # -20% من 42000
            'volatility': 0.10,
            'rsi': 15,
            'price_sma_20': 42000,
            'price_sma_50': 41800,
            'price_change_pct': -0.20
        }
        
        # محاكاة خسارة كبيرة
        self.trading_brain.risk_engine.daily_pnl = -800
        
        risk_metrics = {'daily_pnl': -800}
        market_state = self.trading_brain.state_machine.analyze_market_conditions(features)
        trading_mode = self.trading_brain.state_machine.get_trading_mode(market_state, risk_metrics)
        
        # التحقق من Circuit Breaker
        signal = {'action': 'BUY', 'confidence': 0.8}
        can_trade, reason = self.trading_brain.risk_engine.can_trade(
            signal, features, 100, trading_mode
        )
        
        result = {
            'status': 'PASSED',
            'trading_mode': trading_mode.value,
            'can_trade': can_trade,
            'reason': reason,
            'test_passed': trading_mode.value == 'halt' or not can_trade
        }
        
        # إعادة تعيين
        self.trading_brain.risk_engine.daily_pnl = 0
        
        if result['test_passed']:
            self.logger.info("✅ Flash crash test PASSED - Trading halted or blocked")
        else:
            self.logger.error("❌ Flash crash test FAILED - System did not halt!")
        
        return result
    
    async def test_high_latency(self) -> Dict:
        """اختبار 3: تأخير عالي"""
        
        self.logger.info("🐌 Testing HIGH LATENCY scenario...")
        
        features = {
            'price': 42000,
            'volatility': 0.02,
            'rsi': 50
        }
        
        # محاكاة latency عالي
        high_latency = 1000  # 1 second
        
        signal = {'action': 'BUY', 'confidence': 0.8}
        can_trade, reason = self.trading_brain.risk_engine.can_trade(
            signal, features, high_latency, self.trading_brain.state_machine.current_state
        )
        
        result = {
            'status': 'PASSED',
            'latency': high_latency,
            'can_trade': can_trade,
            'reason': reason,
            'test_passed': not can_trade and 'latency' in reason.lower()
        }
        
        if result['test_passed']:
            self.logger.info("✅ High latency test PASSED - Trade blocked due to latency")
        else:
            self.logger.error("❌ High latency test FAILED - Trade was not blocked!")
        
        return result
    
    async def test_liquidity_crisis(self) -> Dict:
        """اختبار 4: أزمة سيولة"""
        
        self.logger.info("💧 Testing LIQUIDITY CRISIS scenario...")
        
        # محاكاة spread واسع
        features = {
            'price': 42000,
            'volatility': 0.02,
            'rsi': 50,
            'spread': 0.005  # 0.5% spread (عالي جداً)
        }
        
        signal = {'action': 'BUY', 'confidence': 0.8}
        can_trade, reason = self.trading_brain.risk_engine.can_trade(
            signal, features, 100, self.trading_brain.state_machine.current_state
        )
        
        result = {
            'status': 'PASSED',
            'spread': features['spread'],
            'can_trade': can_trade,
            'reason': reason,
            'test_passed': not can_trade and 'spread' in reason.lower()
        }
        
        if result['test_passed']:
            self.logger.info("✅ Liquidity crisis test PASSED - Trade blocked due to wide spread")
        else:
            self.logger.warning("⚠️ Liquidity crisis test WARNING - Trade was not blocked")
        
        return result
    
    async def test_api_failure(self) -> Dict:
        """اختبار 5: فشل API"""
        
        self.logger.info("🔌 Testing API FAILURE scenario...")
        
        # محاكاة فشل في الحصول على features
        features = None
        
        market_state = self.trading_brain.state_machine.analyze_market_conditions(features)
        
        result = {
            'status': 'PASSED',
            'market_state': market_state.value,
            'test_passed': market_state.value == 'maintenance'
        }
        
        if result['test_passed']:
            self.logger.info("✅ API failure test PASSED - System entered maintenance mode")
        else:
            self.logger.error("❌ API failure test FAILED - System did not handle failure!")
        
        return result
    
    def print_report(self):
        """طباعة تقرير الاختبار"""
        
        print("\n" + "="*80)
        print("🔬 STRESS TEST REPORT")
        print("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r.get('test_passed', False))
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result.get('test_passed', False) else "❌ FAILED"
            print(f"\n{test_name}:")
            print(f"  Status: {status}")
            for key, value in result.items():
                if key != 'test_passed':
                    print(f"  {key}: {value}")
        
        print("\n" + "="*80)
        print(f"📊 Summary: {passed_tests}/{total_tests} tests passed")
        print("="*80)

# =============== REAL-TIME MONITOR ===============

class RealTimeMonitor:
    """مراقب حي للنظام"""
    
    def __init__(self, trading_brain, config: MonitorConfig = None):
        self.trading_brain = trading_brain
        self.config = config or MonitorConfig()
        self.logger = logging.getLogger("RealTimeMonitor")
        
        # Metrics history
        self.latency_history = deque(maxlen=self.config.metrics_history_size)
        self.pnl_history = deque(maxlen=self.config.metrics_history_size)
        self.equity_history = deque(maxlen=self.config.metrics_history_size)
        
        # Alerts
        self.alerts = deque(maxlen=100)
        
        # Status
        self.is_monitoring = False
        self.start_time = None
    
    async def start_monitoring(self):
        """بدء المراقبة"""
        
        self.is_monitoring = True
        self.start_time = datetime.now()
        
        self.logger.info("👁️ Real-time monitoring started")
        
        # Start monitoring tasks
        await asyncio.gather(
            self._monitor_loop(),
            self._report_loop()
        )
    
    async def _monitor_loop(self):
        """حلقة المراقبة الرئيسية"""
        
        while self.is_monitoring:
            try:
                # Collect metrics
                status = self.trading_brain.get_status()
                
                # Record metrics
                self.pnl_history.append({
                    'timestamp': datetime.now(),
                    'value': status['daily_pnl']
                })
                
                self.equity_history.append({
                    'timestamp': datetime.now(),
                    'value': status['current_equity']
                })
                
                # Check for alerts
                self._check_alerts(status)
                
                await asyncio.sleep(self.config.monitor_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(1)
    
    async def _report_loop(self):
        """حلقة التقارير الدورية"""
        
        while self.is_monitoring:
            try:
                await asyncio.sleep(self.config.report_interval_seconds)
                self._print_status_report()
                
            except Exception as e:
                self.logger.error(f"Report error: {e}")
    
    def _check_alerts(self, status: Dict):
        """فحص التنبيهات"""
        
        # Check drawdown
        peak = self.trading_brain.risk_engine.peak_equity
        current = status['current_equity']
        drawdown = (peak - current) / peak if peak > 0 else 0
        
        if drawdown > self.config.max_drawdown_critical:
            self._create_alert('CRITICAL', f'Drawdown exceeded {drawdown:.2%}')
        elif drawdown > self.config.max_drawdown_warning:
            self._create_alert('WARNING', f'Drawdown at {drawdown:.2%}')
        
        # Check daily loss
        daily_pnl = status['daily_pnl']
        if daily_pnl < -self.config.daily_loss_critical:
            self._create_alert('CRITICAL', f'Daily loss: ${daily_pnl:.2f}')
        elif daily_pnl < -self.config.daily_loss_warning:
            self._create_alert('WARNING', f'Daily loss: ${daily_pnl:.2f}')
    
    def _create_alert(self, level: str, message: str):
        """إنشاء تنبيه"""
        
        alert = {
            'timestamp': datetime.now(),
            'level': level,
            'message': message
        }
        
        self.alerts.append(alert)
        
        if level == 'CRITICAL':
            self.logger.critical(f"🚨 {message}")
        elif level == 'WARNING':
            self.logger.warning(f"⚠️ {message}")
        else:
            self.logger.info(f"ℹ️ {message}")
    
    def _print_status_report(self):
        """طباعة تقرير الحالة"""
        
        status = self.trading_brain.get_status()
        uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        print("\n" + "="*80)
        print(f"📊 SYSTEM STATUS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print(f"Uptime: {uptime}")
        print(f"Market State: {status['market_state']}")
        print(f"Daily PnL: ${status['daily_pnl']:.2f}")
        print(f"Current Equity: ${status['current_equity']:.2f}")
        print(f"Total Exposure: {status['total_exposure']:.4f} BTC")
        print(f"Trade Count: {status['trade_count']}")
        
        # Recent alerts
        if self.alerts:
            print("\n🚨 Recent Alerts:")
            for alert in list(self.alerts)[-5:]:
                print(f"  [{alert['level']}] {alert['timestamp'].strftime('%H:%M:%S')} - {alert['message']}")
        
        print("="*80)
    
    def stop_monitoring(self):
        """إيقاف المراقبة"""
        self.is_monitoring = False
        self.logger.info("🛑 Monitoring stopped")
    
    def get_metrics_summary(self) -> Dict:
        """الحصول على ملخص المقاييس"""
        
        if not self.pnl_history:
            return {}
        
        pnl_values = [m['value'] for m in self.pnl_history]
        equity_values = [m['value'] for m in self.equity_history]
        
        return {
            'avg_pnl': np.mean(pnl_values),
            'max_pnl': np.max(pnl_values),
            'min_pnl': np.min(pnl_values),
            'current_equity': equity_values[-1] if equity_values else 0,
            'peak_equity': np.max(equity_values) if equity_values else 0,
            'total_alerts': len(self.alerts),
            'critical_alerts': sum(1 for a in self.alerts if a['level'] == 'CRITICAL')
        }

# =============== EXAMPLE USAGE ===============

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Mock components
    from collections import deque
    
    class MockMarketData:
        def __init__(self):
            self.prices = deque(maxlen=200)
            for i in range(100):
                self.prices.append(42000 + np.random.randn() * 100)
    
    # Import Trading Brain
    import sys
    sys.path.append('/home/ubuntu/Vanguard-Alpha')
    
    try:
        from trading_brain import TradingBrain
        
        market_data = MockMarketData()
        brain = TradingBrain(market_data)
        
        # Run stress tests
        tester = StressTester(brain)
        
        async def run_tests():
            results = await tester.run_all_scenarios()
            tester.print_report()
        
        asyncio.run(run_tests())
        
    except ImportError as e:
        print(f"⚠️ Could not import TradingBrain: {e}")
        print("This is a standalone test - TradingBrain will be available after full integration")

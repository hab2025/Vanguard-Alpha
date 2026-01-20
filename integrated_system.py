"""
Vanguard-Alpha Integrated Trading System v2.0
=============================================
النظام الكامل المتكامل - Production Ready

التكامل الكامل:
1. Trading Brain (Master Orchestrator)
2. Advanced AI Engine (Transformer + PPO)
3. Pro Risk Engine (9 فحوصات)
4. Market State Machine (8 حالات)
5. Feature Engineering (20+ ميزة)
6. Stress Testing & Monitoring
7. Professional Backtesting
"""

import asyncio
import logging
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, Optional

# Import components
from trading_brain import TradingBrain, SystemConfig
from advanced_ai_engine import AdvancedAIEngine, AIConfig
from pro_backtesting_engine import ProBacktestEngine, BacktestConfig
from stress_test_monitor import StressTester, RealTimeMonitor, MonitorConfig

# =============== MARKET DATA SERVICE ===============

class LiveMarketData:
    """خدمة بيانات السوق الحية"""
    
    def __init__(self, buffer_size: int = 200):
        self.prices = deque(maxlen=buffer_size)
        self.volumes = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        self.logger = logging.getLogger("LiveMarketData")
    
    def add_tick(self, price: float, volume: float = 0.0, timestamp: Optional[datetime] = None):
        """إضافة tick جديد"""
        self.prices.append(price)
        self.volumes.append(volume)
        self.timestamps.append(timestamp or datetime.now())
    
    def get_latest_price(self) -> Optional[float]:
        """الحصول على آخر سعر"""
        return self.prices[-1] if self.prices else None
    
    def get_price_history(self, n: int = 100) -> list:
        """الحصول على تاريخ الأسعار"""
        return list(self.prices)[-n:]

# =============== EXECUTION SERVICE ===============

class ExecutionService:
    """خدمة تنفيذ الأوامر"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger("ExecutionService")
        self.orders = []
    
    async def execute_order(self, order: Dict):
        """تنفيذ أمر"""
        
        self.logger.info(f"📤 Executing order: {order['action']} {order['quantity']} @ ${order['price']:.2f}")
        
        # في الإنتاج، هنا يتم الاتصال بـ Alpaca API أو Binance API
        # الآن نقوم فقط بتسجيل الأمر
        
        order['execution_time'] = datetime.now()
        order['status'] = 'EXECUTED'
        
        self.orders.append(order)
        
        return order

# =============== DATABASE MANAGER ===============

class DatabaseManager:
    """مدير قاعدة البيانات"""
    
    def __init__(self):
        self.logger = logging.getLogger("DatabaseManager")
        self.trades_log = []
        self.metrics_log = []
    
    def log_trade(self, trade_data: Dict):
        """تسجيل صفقة"""
        self.trades_log.append(trade_data)
        self.logger.debug(f"Trade logged: {trade_data}")
    
    def log_metrics(self, metrics: Dict):
        """تسجيل مقاييس"""
        self.metrics_log.append(metrics)
    
    def get_trades_history(self, limit: int = 100) -> list:
        """الحصول على تاريخ الصفقات"""
        return self.trades_log[-limit:]

# =============== INTEGRATED SYSTEM ===============

class VanguardAlphaSystem:
    """نظام Vanguard-Alpha المتكامل"""
    
    def __init__(self):
        self.logger = logging.getLogger("VanguardAlpha")
        
        # Initialize components
        self.market_data = LiveMarketData()
        self.ai_engine = AdvancedAIEngine(AIConfig())
        self.execution_service = ExecutionService(SystemConfig())
        self.database = DatabaseManager()
        
        # Initialize Trading Brain
        self.brain = TradingBrain(
            market_data=self.market_data,
            ai_engine=self.ai_engine,
            execution_service=self.execution_service,
            database_manager=self.database
        )
        
        # Initialize monitoring
        self.monitor = RealTimeMonitor(self.brain, MonitorConfig())
        
        self.logger.info("="*80)
        self.logger.info("🚀 VANGUARD-ALPHA v2.0 INITIALIZED")
        self.logger.info("="*80)
    
    async def run_live_trading(self, duration_minutes: int = 60):
        """تشغيل التداول الحي"""
        
        self.logger.info(f"▶️ Starting live trading for {duration_minutes} minutes...")
        
        # Start monitoring
        monitor_task = asyncio.create_task(self.monitor.start_monitoring())
        
        # Simulate market data feed
        trading_task = asyncio.create_task(self._trading_loop(duration_minutes))
        
        await asyncio.gather(monitor_task, trading_task)
    
    async def _trading_loop(self, duration_minutes: int):
        """حلقة التداول الرئيسية"""
        
        start_time = datetime.now()
        tick_count = 0
        
        while (datetime.now() - start_time).seconds < duration_minutes * 60:
            try:
                # Simulate incoming tick (في الإنتاج، يأتي من WebSocket)
                current_price = 42000 + np.random.randn() * 100
                self.market_data.add_tick(current_price, volume=1000)
                
                # Process tick through Trading Brain
                tick_data = {
                    'price': current_price,
                    'latency': np.random.randint(50, 150),
                    'timestamp': datetime.now()
                }
                
                order = await self.brain.process_tick(tick_data)
                
                if order:
                    self.logger.info(f"✅ Order placed: {order}")
                
                tick_count += 1
                
                # Simulate tick rate (في الإنتاج، يعتمد على WebSocket)
                await asyncio.sleep(0.1)  # 10 ticks per second
                
            except Exception as e:
                self.logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(1)
        
        self.logger.info(f"⏹️ Trading stopped after {tick_count} ticks")
        self.monitor.stop_monitoring()
    
    async def run_stress_tests(self):
        """تشغيل اختبارات الضغط"""
        
        self.logger.info("🔬 Running comprehensive stress tests...")
        
        tester = StressTester(self.brain)
        results = await tester.run_all_scenarios()
        tester.print_report()
        
        return results
    
    def run_backtest(self, price_data, features_data):
        """تشغيل Backtesting"""
        
        self.logger.info("📊 Running professional backtest...")
        
        # Create backtest engine with AI strategy
        backtest_config = BacktestConfig(
            initial_capital=10000,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        engine = ProBacktestEngine(self.ai_engine, backtest_config)
        
        # Run backtest
        metrics = engine.run_backtest(price_data, features_data)
        
        # Print report
        engine.print_report()
        
        # Monte Carlo simulation
        mc_results = engine.monte_carlo_simulation()
        
        if mc_results:
            print("\n" + "="*80)
            print("🎲 MONTE CARLO SIMULATION RESULTS")
            print("="*80)
            for key, value in mc_results.items():
                if isinstance(value, float):
                    print(f"{key:<30}: ${value:,.2f}")
                else:
                    print(f"{key:<30}: {value}")
            print("="*80)
        
        # Plot results
        engine.plot_results()
        
        return metrics
    
    def get_system_status(self) -> Dict:
        """الحصول على حالة النظام"""
        
        brain_status = self.brain.get_status()
        monitor_metrics = self.monitor.get_metrics_summary()
        
        return {
            'brain': brain_status,
            'monitor': monitor_metrics,
            'ai_engine': {
                'training_steps': self.ai_engine.training_steps,
                'replay_buffer_size': len(self.ai_engine.replay_buffer),
                'device': self.ai_engine.config.device
            },
            'database': {
                'trades_count': len(self.database.trades_log),
                'metrics_count': len(self.database.metrics_log)
            }
        }
    
    def print_status(self):
        """طباعة حالة النظام"""
        
        status = self.get_system_status()
        
        print("\n" + "="*80)
        print("📊 VANGUARD-ALPHA SYSTEM STATUS")
        print("="*80)
        
        print("\n🧠 Trading Brain:")
        for key, value in status['brain'].items():
            print(f"  {key}: {value}")
        
        print("\n🤖 AI Engine:")
        for key, value in status['ai_engine'].items():
            print(f"  {key}: {value}")
        
        print("\n👁️ Monitor:")
        for key, value in status['monitor'].items():
            print(f"  {key}: {value}")
        
        print("\n💾 Database:")
        for key, value in status['database'].items():
            print(f"  {key}: {value}")
        
        print("="*80)

# =============== MAIN ENTRY POINT ===============

async def main():
    """نقطة الدخول الرئيسية"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create system
    system = VanguardAlphaSystem()
    
    # Menu
    print("\n" + "="*80)
    print("🚀 VANGUARD-ALPHA v2.0 - INTEGRATED TRADING SYSTEM")
    print("="*80)
    print("\nSelect mode:")
    print("1. Run Stress Tests")
    print("2. Run Live Trading (Simulated)")
    print("3. Run Backtest")
    print("4. Show System Status")
    print("5. Exit")
    print("="*80)
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        # Stress tests
        await system.run_stress_tests()
    
    elif choice == '2':
        # Live trading
        duration = int(input("Enter duration in minutes (default 5): ") or "5")
        await system.run_live_trading(duration)
    
    elif choice == '3':
        # Backtest
        print("\n📥 Loading sample data...")
        import pandas as pd
        
        # Generate sample data
        dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
        price_data = pd.DataFrame({
            'Close': 42000 + np.cumsum(np.random.randn(1000) * 100)
        }, index=dates)
        
        features_data = pd.DataFrame(np.random.randn(1000, 20))
        
        system.run_backtest(price_data, features_data)
    
    elif choice == '4':
        # Status
        system.print_status()
    
    else:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 System shutdown requested...")
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        import traceback
        traceback.print_exc()

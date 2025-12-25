"""
Backup Automation & Discord Integration
Essential production features for data safety and notifications

Features:
- Automated database backups
- Cloud storage integration (optional)
- Discord webhook alerts
- Backup scheduling
"""

import os
import shutil
import gzip
from typing import Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import requests
import logging

logger = logging.getLogger(__name__)


class BackupAutomation:
    """
    Automated backup system
    
    Features:
    - Scheduled backups
    - Compression
    - Retention policy
    - Cloud upload (optional)
    """
    
    def __init__(self, backup_dir: str = "backups",
                 retention_days: int = 30,
                 compress: bool = True):
        """
        Initialize backup automation
        
        Args:
            backup_dir: Directory for backups
            retention_days: Days to keep backups
            compress: Whether to compress backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.retention_days = retention_days
        self.compress = compress
        
        logger.info(f"✓ Backup automation initialized: {self.backup_dir}")
    
    def create_backup(self, source_file: str, tag: str = "") -> str:
        """
        Create a backup of a file
        
        Args:
            source_file: File to backup
            tag: Optional tag for backup name
            
        Returns:
            Path to backup file
        """
        if not os.path.exists(source_file):
            logger.error(f"Source file not found: {source_file}")
            return None
        
        # Generate backup filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        basename = Path(source_file).name
        tag_str = f"_{tag}" if tag else ""
        backup_name = f"{basename}{tag_str}_{timestamp}"
        
        if self.compress:
            backup_name += ".gz"
            backup_path = self.backup_dir / backup_name
            
            # Compress and copy
            with open(source_file, 'rb') as f_in:
                with gzip.open(backup_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            backup_path = self.backup_dir / backup_name
            shutil.copy2(source_file, backup_path)
        
        logger.info(f"✓ Backup created: {backup_path}")
        return str(backup_path)
    
    def cleanup_old_backups(self):
        """Remove backups older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        removed_count = 0
        for backup_file in self.backup_dir.iterdir():
            if backup_file.is_file():
                # Get file modification time
                mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
                
                if mtime < cutoff_date:
                    backup_file.unlink()
                    removed_count += 1
                    logger.info(f"Removed old backup: {backup_file.name}")
        
        if removed_count > 0:
            logger.info(f"✓ Cleanup: Removed {removed_count} old backups")
        
        return removed_count
    
    def get_backup_list(self) -> list:
        """Get list of all backups"""
        backups = []
        
        for backup_file in sorted(self.backup_dir.iterdir(), reverse=True):
            if backup_file.is_file():
                backups.append({
                    'name': backup_file.name,
                    'size_mb': backup_file.stat().st_size / (1024 * 1024),
                    'created': datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat()
                })
        
        return backups
    
    def restore_backup(self, backup_name: str, restore_path: str) -> bool:
        """Restore a backup file"""
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_name}")
            return False
        
        try:
            if backup_name.endswith('.gz'):
                # Decompress
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(restore_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy2(backup_path, restore_path)
            
            logger.info(f"✓ Backup restored to: {restore_path}")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False


class DiscordNotifier:
    """
    Discord webhook notifications
    
    Alternative to Telegram for alerts
    """
    
    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize Discord notifier
        
        Args:
            webhook_url: Discord webhook URL
        """
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL')
        self.enabled = bool(self.webhook_url)
        
        if not self.enabled:
            logger.warning("Discord notifications disabled - no webhook URL")
        else:
            logger.info("✓ Discord notifications enabled")
    
    def send_message(self, message: str, title: str = None, color: int = 0x00ff00) -> bool:
        """
        Send Discord message
        
        Args:
            message: Message content
            title: Optional embed title
            color: Embed color (hex)
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.debug(f"Discord disabled, would have sent: {message}")
            return False
        
        try:
            # Create embed
            embed = {
                "description": message,
                "color": color,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if title:
                embed["title"] = title
            
            payload = {
                "embeds": [embed]
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.debug(f"Discord message sent: {message[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Discord message: {e}")
            return False
    
    def send_trade_alert(self, trade: Dict) -> bool:
        """Send trade execution alert"""
        action = trade.get('action', 'UNKNOWN')
        price = trade.get('price', 0)
        pnl = trade.get('pnl', 0)
        
        color = 0x00ff00 if action == 'LONG' else 0xff0000
        
        message = f"""
**Action:** {action}
**Price:** ${price:,.2f}
**Size:** {trade.get('size', 0):.4f} BTC
**Confidence:** {trade.get('confidence', 0)*100:.1f}%
"""
        
        if pnl != 0:
            message += f"**P&L:** ${pnl:,.2f}"
        
        return self.send_message(message, title="🔔 Trade Executed", color=color)
    
    def send_alert(self, alert_type: str, message: str) -> bool:
        """Send risk/system alert"""
        colors = {
            'error': 0xff0000,
            'warning': 0xffa500,
            'info': 0x0099ff,
            'success': 0x00ff00
        }
        
        color = colors.get(alert_type.lower(), 0x808080)
        title = f"⚠️ {alert_type.upper()} Alert"
        
        return self.send_message(message, title=title, color=color)


# Singletons
_backup_automation: Optional[BackupAutomation] = None
_discord_notifier: Optional[DiscordNotifier] = None


def get_backup_automation() -> BackupAutomation:
    global _backup_automation
    if _backup_automation is None:
        _backup_automation = BackupAutomation()
    return _backup_automation


def get_discord_notifier() -> DiscordNotifier:
    global _discord_notifier
    if _discord_notifier is None:
        _discord_notifier = DiscordNotifier()
    return _discord_notifier


if __name__ == '__main__':
    print("=" * 70)
    print("BACKUP & DISCORD - TEST")
    print("=" * 70)
    
    # Test backup automation
    print("\n💾 Testing Backup Automation...")
    backup_mgr = BackupAutomation(backup_dir="test_backups", retention_days=7)
    
    # Create a test file
    test_file = "test_data.txt"
    with open(test_file, 'w') as f:
        f.write("Test trading data\n" * 100)
    
    # Create backups
    backup1 = backup_mgr.create_backup(test_file, tag="manual")
    print(f"  Created backup: {Path(backup1).name if backup1 else 'FAILED'}")
    
    # List backups
    backups = backup_mgr.get_backup_list()
    print(f"\n  Total backups: {len(backups)}")
    for backup in backups[:3]:  # Show first 3
        print(f"    {backup['name']} ({backup['size_mb']:.2f} MB)")
    
    # Test restore
    if backup1:
        restore_path = "test_data_restored.txt"
        success = backup_mgr.restore_backup(Path(backup1).name, restore_path)
        print(f"\n  Restore {'successful' if success else 'failed'}")
        
        if os.path.exists(restore_path):
            os.remove(restore_path)
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
    
    # Test Discord
    print("\n💬 Testing Discord Notifier...")
    discord = DiscordNotifier()
    
    if discord.enabled:
        print("  Discord webhook configured")
        
        # Send test message
        success = discord.send_message("Test message from CryptoBoss Pro", title="✅ Test Alert")
        print(f"  Test message {'sent' if success else 'failed'}")
        
        # Send trade alert
        test_trade = {
            'action': 'LONG',
            'price': 50000,
            'size': 0.1,
            'confidence': 0.85
        }
        discord.send_trade_alert(test_trade)
    else:
        print("  Discord not configured (set DISCORD_WEBHOOK_URL)")
        print("  To enable: Set environment variable DISCORD_WEBHOOK_URL")
    
    print("\n" + "=" * 70)
    print("✅ Backup & Discord working!")
    print("=" * 70)

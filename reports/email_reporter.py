"""
Email Reporter - Automated Performance Reports
Sends daily/weekly performance summaries via email

Setup: Set SMTP credentials in environment variables or config
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EmailReporter:
    """
    Professional email reporting system
    
    Features:
    - Daily performance summaries
    - Weekly P&L reports
    - Risk alerts
    - HTML-formatted emails
    """
    
    def __init__(self, 
                 smtp_server: Optional[str] = None,
                 smtp_port: Optional[int] = None,
                 sender_email: Optional[str] = None,
                 sender_password: Optional[str] = None,
                 recipient_email: Optional[str] = None):
        """
        Initialize email reporter
        
        Args:
            smtp_server: SMTP server (e.g., 'smtp.gmail.com')
            smtp_port: SMTP port (e.g., 587 for TLS)
            sender_email: Sender email address
            sender_password: Email password or app password
            recipient_email: Recipient email address
        """
        self.smtp_server = smtp_server or os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = smtp_port or int(os.getenv('SMTP_PORT', '587'))
        self.sender_email = sender_email or os.getenv('SENDER_EMAIL')
        self.sender_password = sender_password or os.getenv('SENDER_PASSWORD')
        self.recipient_email = recipient_email or os.getenv('RECIPIENT_EMAIL')
        
        self.enabled = bool(self.sender_email and self.sender_password and self.recipient_email)
        
        if not self.enabled:
            logger.warning("Email reports disabled - missing credentials")
            logger.info("Set SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL environment variables")
        else:
            logger.info(f"✓ Email reports enabled: {self.recipient_email}")
    
    def send_email(self, subject: str, body_html: str) -> bool:
        """
        Send HTML email
        
        Args:
            subject: Email subject
            body_html: HTML body content
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.debug(f"Email disabled, would have sent: {subject}")
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            
            # Attach HTML body
            html_part = MIMEText(body_html, 'html')
            msg.attach(html_part)
            
            # Send via SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Email sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def send_daily_report(self, metrics: Dict) -> bool:
        """
        Send daily performance report
        
        Args:
            metrics: Performance metrics dictionary
        """
        equity = metrics.get('equity', 0)
        total_pnl = metrics.get('total_pnl', 0)
        daily_pnl = metrics.get('daily_pnl', 0)
        win_rate = metrics.get('win_rate', 0)
        total_trades = metrics.get('total_trades', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown',  0)
        
        # Determine color based on daily P&L
        pnl_color = '#10b981' if daily_pnl >= 0 else '#ef4444'
        
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background: #1e293b; color: white; padding: 20px; }}
                .metric-card {{ background: #f1f5f9; padding: 15px; margin: 10px 0; border-radius: 8px; }}
                .metric-label {{ color: #64748b; font-size: 14px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #0f172a; }}
                .positive {{ color: #10b981; }}
                .negative {{ color: #ef4444; }}
                .footer {{ color: #94a3b8; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 CryptoBoss Pro - Daily Report</h1>
                <p>{datetime.now().strftime('%A, %B %d, %Y')}</p>
            </div>
            
            <div style="padding: 20px;">
                <h2>Performance Summary</h2>
                
                <div class="metric-card">
                    <div class="metric-label">Portfolio Value</div>
                    <div class="metric-value">${equity:,.2f}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Today's P&L</div>
                    <div class="metric-value" style="color: {pnl_color};">
                        ${daily_pnl:+,.2f}
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Total P&L</div>
                    <div class="metric-value" style="color: {'#10b981' if total_pnl >= 0 else '#ef4444'};">
                        ${total_pnl:+,.2f}
                    </div>
                </div>
                
                <h2>Trading Statistics</h2>
                
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">Total Trades</td>
                        <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; text-align: right;"><strong>{total_trades}</strong></td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">Win Rate</td>
                        <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; text-align: right;"><strong>{win_rate*100:.1f}%</strong></td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">Sharpe Ratio</td>
                        <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; text-align: right;"><strong>{sharpe:.2f}</strong></td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">Max Drawdown</td>
                        <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; text-align: right;"><strong>{max_dd*100:.2f}%</strong></td>
                    </tr>
                </table>
                
                <div class="footer">
                    <p>This is an automated report from CryptoBoss Pro Trading Bot</p>
                    <p>Generated at {datetime.now().strftime('%H:%M:%S UTC')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        subject = f"📊 Daily Report - ${daily_pnl:+,.2f} - {datetime.now().strftime('%Y-%m-%d')}"
        
        return self.send_email(subject, html_body)
    
    def send_alert(self, alert_type: str, message: str) -> bool:
        """
        Send risk alert email
        
        Args:
            alert_type: Type of alert (e.g., 'drawdown', 'circuit_breaker')
            message: Alert message
        """
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="background: #fef2f2; border-left: 4px solid #ef4444; padding: 20px; margin: 20px;">
                <h2 style="color: #991b1b; margin-top: 0;">⚠️ Risk Alert: {alert_type.upper()}</h2>
                <p style="font-size: 16px;">{message}</p>
                <hr style="border-color: #fca5a5;">
                <p style="color: #64748b; font-size: 14px;">
                    Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}<br>
                    Bot: CryptoBoss Pro
                </p>
            </div>
        </body>
        </html>
        """
        
        subject = f"🚨 Alert: {alert_type.upper()}"
        
        return self.send_email(subject, html_body)
    
    def test_connection(self) -> bool:
        """Test email configuration"""
        if not self.enabled:
            print("❌ Email not enabled - set credentials")
            return False
        
        test_html = """
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #10b981;">✅ Email Test Successful!</h2>
            <p>Your CryptoBoss Pro email reporting is configured correctly.</p>
            <p style="color: #64748b; font-size: 14px;">
                This is a test message sent at {time}
            </p>
        </body>
        </html>
        """.format(time=datetime.now().strftime('%H:%M:%S'))
        
        result = self.send_email("✅ CryptoBoss Pro - Email Test", test_html)
        
        if result:
            print("✅ Test email sent successfully")
        else:
            print("❌ Failed to send test email")
        
        return result


# Singleton
_email_reporter: Optional[EmailReporter] = None


def get_email_reporter() -> EmailReporter:
    """Get singleton email reporter"""
    global _email_reporter
    if _email_reporter is None:
        _email_reporter = EmailReporter()
    return _email_reporter


if __name__ == '__main__':
    print("=" * 70)
    print("EMAIL REPORTER - SETUP & TEST")
    print("=" * 70)
    
    print("\n📧 Email Setup Instructions:")
    print("1. For Gmail: Enable 2-factor auth, create app password")
    print("2. Set environment variables:")
    print("   export SENDER_EMAIL='your@gmail.com'")
    print("   export SENDER_PASSWORD='your_app_password'")
    print("   export RECIPIENT_EMAIL='recipient@email.com'")
    print("   export SMTP_SERVER='smtp.gmail.com'  # Optional, defaults to Gmail")
    print("   export SMTP_PORT='587'  # Optional, defaults to 587")
    
    # Initialize
    reporter = EmailReporter()
    
    if reporter.enabled:
        print("\n✅ Email configuration found!")
        print(f"Sender: {reporter.sender_email}")
        print(f"Recipient: {reporter.recipient_email}")
        
        print("\n🧪 Running connection test...")
        if reporter.test_connection():
            print("\n📊 Sending sample daily report...")
            
            sample_metrics = {
                'equity': 10234.56,
                'total_pnl': 234.56,
                'daily_pnl': 45.67,
                'win_rate': 0.68,
                'total_trades': 47,
                'sharpe_ratio': 2.15,
                'max_drawdown': 0.05
            }
            
            reporter.send_daily_report(sample_metrics)
            print("✅ Sample report sent!")
    else:
        print("\n⚠️  Email not configured (running in disabled mode)")
        print("Set environment variables to enable email reports")
    
    print("\n" + "=" * 70)

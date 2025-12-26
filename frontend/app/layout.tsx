import './globals.css';
import type { Metadata } from 'next';
import { JetBrains_Mono } from 'next/font/google';

const jetbrainsMono = JetBrains_Mono({ subsets: ['latin'] });

export const metadata: Metadata = {
    title: 'CryptoBoss Pro - Advanced Trading System',
    description: 'Professional crypto trading system with systematic strategies',
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en">
            <body className={jetbrainsMono.className}>{children}</body>
        </html>
    );
}

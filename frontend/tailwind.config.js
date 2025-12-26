/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        './pages/**/*.{js,ts,jsx,tsx,mdx}',
        './components/**/*.{js,ts,jsx,tsx,mdx}',
        './app/**/*.{js,ts,jsx,tsx,mdx}',
    ],
    theme: {
        extend: {
            colors: {
                // Professional dark theme
                bg: {
                    primary: '#0f1419',
                    secondary: '#16181d',
                    tertiary: '#1e2128',
                },
                accent: {
                    green: '#00c853',
                    red: '#ff1744',
                    blue: '#2979ff',
                    yellow: '#ffd600',
                },
                text: {
                    primary: '#e8eaed',
                    secondary: '#9aa0a6',
                    muted: '#5f6368',
                },
                border: '#2e3338',
            },
            fontFamily: {
                mono: ['JetBrains Mono', 'monospace'],
            },
        },
    },
    plugins: [],
};

interface CardProps {
    children: React.ReactNode;
    title?: string;
    subtitle?: string;
    className?: string;
    noPadding?: boolean;
    interactive?: boolean;
}

export function Card({ children, title, subtitle, className = '', noPadding = false, interactive = true }: CardProps) {
    return (
        <div
            className={`
                bg-[#1a1f26] rounded-lg border border-[#2d3640] 
                shadow-[0_1px_3px_rgba(0,0,0,0.12),0_1px_2px_rgba(0,0,0,0.24)]
                ${interactive ? 'transition-all duration-200 hover:shadow-[0_3px_6px_rgba(0,0,0,0.16),0_3px_6px_rgba(0,0,0,0.23)] hover:border-[#3d4650]' : ''}
                ${className}
            `}
        >
            {(title || subtitle) && (
                <div className="px-4 py-3 border-b border-[#2d3640]">
                    {title && <h3 className="text-sm font-medium text-[#e7e9ea]">{title}</h3>}
                    {subtitle && <p className="text-xs text-[#8b98a5] mt-0.5">{subtitle}</p>}
                </div>
            )}
            <div className={noPadding ? '' : 'p-4'}>
                {children}
            </div>
        </div>
    );
}

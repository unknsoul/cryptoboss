interface CardProps {
    children: React.ReactNode;
    title?: string;
    subtitle?: string;
    className?: string;
    noPadding?: boolean;
}

export function Card({ children, title, subtitle, className = '', noPadding = false }: CardProps) {
    return (
        <div className={`bg-[#1a1f26] rounded-lg border border-[#2d3640] ${className}`}>
            {(title || subtitle) && (
                <div className="px-4 py-3 border-b border-[#2d3640]">
                    {title && <h3 className="text-sm font-medium text-white">{title}</h3>}
                    {subtitle && <p className="text-xs text-[#8b98a5] mt-0.5">{subtitle}</p>}
                </div>
            )}
            <div className={noPadding ? '' : 'p-4'}>
                {children}
            </div>
        </div>
    );
}

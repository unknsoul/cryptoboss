type BadgeVariant = 'default' | 'success' | 'warning' | 'danger' | 'info' | 'neutral';

interface BadgeProps {
    children: React.ReactNode;
    variant?: BadgeVariant;
    size?: 'sm' | 'md';
}

const variantStyles: Record<BadgeVariant, string> = {
    default: 'bg-[#242b33] text-[#e7e9ea]',
    success: 'bg-green-500/20 text-green-400',
    warning: 'bg-yellow-500/20 text-yellow-400',
    danger: 'bg-red-500/20 text-red-400',
    info: 'bg-blue-500/20 text-blue-400',
    neutral: 'bg-[#2d3640] text-[#8b98a5]',
};

export function Badge({ children, variant = 'default', size = 'sm' }: BadgeProps) {
    const sizeStyles = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm';

    return (
        <span className={`inline-flex items-center rounded-full font-medium ${variantStyles[variant]} ${sizeStyles}`}>
            {children}
        </span>
    );
}

/**
 * Badge Component
 * Uses muted colors per CryptoBoss design spec
 * - No celebratory/bright colors
 * - Professional trading control room aesthetic
 */

type BadgeVariant = 'default' | 'success' | 'warning' | 'danger' | 'info' | 'neutral' | 'accent';

interface BadgeProps {
    children: React.ReactNode;
    variant?: BadgeVariant;
    size?: 'sm' | 'md';
}

// Muted color palette per spec
const variantStyles: Record<BadgeVariant, string> = {
    default: 'bg-[#242b33] text-[#e7e9ea]',
    success: 'bg-[rgba(74,146,104,0.15)] text-[#4a9268]',       // Soft green
    warning: 'bg-[rgba(196,160,82,0.15)] text-[#c4a052]',       // Amber
    danger: 'bg-[rgba(166,84,84,0.15)] text-[#a65454]',         // Muted red
    info: 'bg-[rgba(91,122,157,0.15)] text-[#5b7a9d]',          // Muted blue
    neutral: 'bg-[#2d3640] text-[#8b98a5]',
    accent: 'bg-[rgba(91,122,157,0.15)] text-[#5b7a9d]',        // Muted blue accent
};

export function Badge({ children, variant = 'default', size = 'sm' }: BadgeProps) {
    const sizeStyles = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm';

    return (
        <span className={`inline-flex items-center rounded font-medium ${variantStyles[variant]} ${sizeStyles}`}>
            {children}
        </span>
    );
}

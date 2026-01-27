type StatusDotVariant = 'success' | 'warning' | 'danger' | 'neutral' | 'info';

interface StatusDotProps {
    status: StatusDotVariant;
    pulse?: boolean;
    size?: 'sm' | 'md' | 'lg';
    label?: string;
}

const statusColors: Record<StatusDotVariant, string> = {
    success: 'bg-green-500',
    warning: 'bg-yellow-500',
    danger: 'bg-red-500',
    neutral: 'bg-gray-500',
    info: 'bg-blue-500',
};

export function StatusDot({ status, pulse = false, size = 'md', label }: StatusDotProps) {
    const sizeStyles = {
        sm: 'w-2 h-2',
        md: 'w-2.5 h-2.5',
        lg: 'w-3 h-3',
    };

    return (
        <div className="flex items-center gap-2">
            <div
                className={`rounded-full ${statusColors[status]} ${sizeStyles[size]} ${pulse ? 'animate-pulse' : ''
                    }`}
            />
            {label && <span className="text-sm text-[#8b98a5]">{label}</span>}
        </div>
    );
}

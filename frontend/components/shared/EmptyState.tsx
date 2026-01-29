'use client';

interface EmptyStateProps {
    icon?: React.ReactNode;
    title: string;
    description?: string;
    action?: React.ReactNode;
}

export function EmptyState({
    icon,
    title,
    description,
    action
}: EmptyStateProps) {
    const defaultIcon = (
        <svg
            className="w-8 h-8 text-[#6b7280]"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
        >
            <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"
            />
        </svg>
    );

    return (
        <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="w-16 h-16 mb-4 rounded-full bg-[#2d3640] flex items-center justify-center">
                {icon || defaultIcon}
            </div>
            <p className="text-[#e7e9ea] font-medium mb-1">{title}</p>
            {description && (
                <p className="text-[#6b7280] text-sm mb-4 max-w-sm">{description}</p>
            )}
            {action}
        </div>
    );
}

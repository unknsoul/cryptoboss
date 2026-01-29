'use client';

interface SkeletonProps {
    className?: string;
    variant?: 'text' | 'title' | 'value' | 'circle' | 'rect';
    width?: string;
    height?: string;
}

export function Skeleton({
    className = '',
    variant = 'text',
    width,
    height
}: SkeletonProps) {
    const variantClasses = {
        text: 'h-4 w-full',
        title: 'h-6 w-3/4',
        value: 'h-8 w-20',
        circle: 'h-10 w-10 rounded-full',
        rect: 'h-24 w-full'
    };

    return (
        <div
            className={`
                animate-pulse bg-gradient-to-r from-[#242b33] via-[#2d3640] to-[#242b33]
                bg-[length:200%_100%] rounded
                ${variantClasses[variant]}
                ${className}
            `}
            style={{
                width: width || undefined,
                height: height || undefined,
                animation: 'shimmer 1.5s ease-in-out infinite'
            }}
        />
    );
}

export function SkeletonCard({ rows = 3 }: { rows?: number }) {
    return (
        <div className="bg-[#1a1f26] rounded-lg border border-[#2d3640] p-4">
            <div className="space-y-4">
                <Skeleton variant="title" width="40%" />
                {[...Array(rows)].map((_, i) => (
                    <div key={i} className="flex justify-between items-center">
                        <Skeleton variant="text" width="30%" />
                        <Skeleton variant="value" width="20%" />
                    </div>
                ))}
                <Skeleton variant="text" className="h-1.5 rounded-full" />
            </div>
        </div>
    );
}

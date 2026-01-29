'use client';

interface PageHeaderProps {
    title: string;
    description?: string;
    children?: React.ReactNode;
}

export function PageHeader({ title, description, children }: PageHeaderProps) {
    return (
        <div className="mb-8 flex items-start justify-between">
            <div>
                <h1 className="text-2xl font-semibold text-[#e7e9ea] tracking-tight">{title}</h1>
                {description && (
                    <p className="text-[#8b98a5] text-sm mt-1">{description}</p>
                )}
            </div>
            {children && (
                <div className="flex items-center gap-3">
                    {children}
                </div>
            )}
        </div>
    );
}

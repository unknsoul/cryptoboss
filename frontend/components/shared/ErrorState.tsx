'use client';

interface ErrorStateProps {
    message?: string;
    onRetry?: () => void;
}

export function ErrorState({
    message = 'Failed to load data',
    onRetry
}: ErrorStateProps) {
    return (
        <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="w-16 h-16 mb-4 rounded-full bg-[#a65454]/10 flex items-center justify-center">
                <svg
                    className="w-8 h-8 text-[#a65454]"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                >
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                    />
                </svg>
            </div>
            <p className="text-[#a65454] font-medium mb-1">Connection Error</p>
            <p className="text-[#6b7280] text-sm mb-4">{message}</p>
            {onRetry && (
                <button
                    onClick={onRetry}
                    className="px-4 py-2 rounded-md text-sm font-medium bg-[#5b7a9d] text-white hover:bg-[#6b8aad] transition-colors"
                >
                    Try Again
                </button>
            )}
        </div>
    );
}

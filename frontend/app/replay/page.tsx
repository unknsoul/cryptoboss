import { Card } from '../../components/shared/Card';
import { Badge } from '../../components/shared/Badge';
import { PageHeader } from '../../components/layout/PageHeader';

const mockSessions = [
    { id: 'replay_001', date: '2026-01-27', duration: '4h 30m', decisions: 45, mismatches: 0 },
    { id: 'replay_002', date: '2026-01-26', duration: '8h 15m', decisions: 92, mismatches: 2 },
    { id: 'replay_003', date: '2026-01-25', duration: '6h 00m', decisions: 68, mismatches: 0 },
];

export default function ReplayPage() {
    return (
        <div>
            <PageHeader
                title="Replay & Analysis"
                description="Post-mortem and validation tools"
            />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Session Selector */}
                <Card title="Replay Sessions" className="lg:col-span-2">
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="border-b border-[#2d3640]">
                                    <th className="text-left text-xs text-[#8b98a5] font-medium py-3 px-4">Session ID</th>
                                    <th className="text-left text-xs text-[#8b98a5] font-medium py-3 px-4">Date</th>
                                    <th className="text-left text-xs text-[#8b98a5] font-medium py-3 px-4">Duration</th>
                                    <th className="text-right text-xs text-[#8b98a5] font-medium py-3 px-4">Decisions</th>
                                    <th className="text-right text-xs text-[#8b98a5] font-medium py-3 px-4">Mismatches</th>
                                    <th className="text-right text-xs text-[#8b98a5] font-medium py-3 px-4">Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {mockSessions.map((session) => (
                                    <tr key={session.id} className="border-b border-[#2d3640] hover:bg-[#1a1f26] cursor-pointer">
                                        <td className="py-3 px-4 text-white font-mono text-sm">{session.id}</td>
                                        <td className="py-3 px-4 text-[#8b98a5]">{session.date}</td>
                                        <td className="py-3 px-4 text-white">{session.duration}</td>
                                        <td className="py-3 px-4 text-right text-white">{session.decisions}</td>
                                        <td className="py-3 px-4 text-right">
                                            <span className={session.mismatches > 0 ? 'text-red-400' : 'text-green-400'}>
                                                {session.mismatches}
                                            </span>
                                        </td>
                                        <td className="py-3 px-4 text-right">
                                            <Badge variant={session.mismatches === 0 ? 'success' : 'warning'}>
                                                {session.mismatches === 0 ? 'MATCH' : 'DIVERGED'}
                                            </Badge>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </Card>

                {/* Comparison Panel */}
                <Card title="Live vs Replay Comparison">
                    <div className="text-center py-8 text-[#8b98a5]">
                        <p>Select a session to view comparison</p>
                    </div>
                </Card>

                {/* Mismatch Alerts */}
                <Card title="Mismatch Alerts">
                    <div className="text-center py-8 text-[#8b98a5]">
                        <p>No mismatches in current session</p>
                    </div>
                </Card>
            </div>
        </div>
    );
}

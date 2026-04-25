export function unwrapApiData<T = any>(payload: any): T {
    if (payload && typeof payload === 'object' && 'data' in payload) {
        return payload.data as T;
    }
    return payload as T;
}

export async function readApiData<T = any>(response: Response): Promise<T> {
    const payload = await response.json();
    return unwrapApiData<T>(payload);
}

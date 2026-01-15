import React, { useState } from 'react';

export default function RAGChat() {
    const [query, setQuery] = useState('');
    const [response, setResponse] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSearch = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setResponse(null);

        try {
            const res = await fetch('http://localhost:8000/rag', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query }),
            });

            if (!res.ok) throw new Error("API Failure");

            const data = await res.json();
            setResponse(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-100 h-full flex flex-col">
            <h2 className="text-xl font-bold mb-4 text-slate-800 flex items-center">
                🤖 Assistant
            </h2>

            <div className="flex-1 overflow-y-auto mb-4 space-y-4 min-h-[300px] max-h-[500px] pr-2 custom-scrollbar">
                {response ? (
                    <div className="space-y-4">
                        <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
                            <p className="text-slate-800 leading-relaxed whitespace-pre-wrap">{response.answer}</p>
                        </div>

                        <div className="bg-blue-50 p-3 rounded-md text-xs border border-blue-100">
                            <h4 className="font-semibold text-blue-800 mb-2">Sources/Context:</h4>
                            <ul className="list-disc pl-4 space-y-1 text-blue-700">
                                {response.context.map((ctx, idx) => (
                                    <li key={idx}>{ctx}</li>
                                ))}
                            </ul>
                        </div>
                    </div>
                ) : (
                    <div className="flex items-center justify-center h-full text-gray-400 text-sm italic">
                        Ask query about urban issues...
                    </div>
                )}
            </div>

            <form onSubmit={handleSearch} className="mt-auto">
                <div className="flex gap-2">
                    <input
                        type="text"
                        className="flex-1 p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500"
                        placeholder="e.g. Traffic in downtown..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        required
                    />
                    <button
                        type="submit"
                        disabled={loading}
                        className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 disabled:opacity-50"
                    >
                        {loading ? '...' : 'Send'}
                    </button>
                </div>
            </form>
            {error && <p className="text-red-500 text-xs mt-2">{error}</p>}
        </div>
    );
}

import React, { useState } from 'react';

export default function ComplaintForm() {
    const [text, setText] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const response = await fetch('http://localhost:8000/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text }),
            });

            if (!response.ok) throw new Error("Failed to connect to API");

            const data = await response.json();
            setResult(data.sentiment);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const getSentimentColor = (sentiment) => {
        if (sentiment === 'good') return 'bg-green-100 text-green-800 border-green-200';
        if (sentiment === 'bad') return 'bg-red-100 text-red-800 border-red-200';
        return 'bg-gray-100 text-gray-800 border-gray-200';
    };

    return (
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-100">
            <h2 className="text-xl font-bold mb-4 text-slate-800 flex items-center">
                📊 Complaint Analysis
            </h2>
            <form onSubmit={handleSubmit}>
                <textarea
                    className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent h-32 resize-none"
                    placeholder="Describe the issue..."
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    required
                />
                <button
                    type="submit"
                    disabled={loading}
                    className="mt-3 w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition disabled:opacity-50 font-medium"
                >
                    {loading ? 'Analyzing...' : 'Analyze Sentiment'}
                </button>
            </form>

            {error && (
                <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md text-sm">
                    {error}
                </div>
            )}

            {result && (
                <div className={`mt-4 p-4 rounded-md border text-center animate-fade-in ${getSentimentColor(result)}`}>
                    <p className="text-sm uppercase tracking-wide font-semibold">Detected Sentiment</p>
                    <p className="text-2xl font-bold mt-1 capitalize">{result}</p>
                </div>
            )}
        </div>
    );
}

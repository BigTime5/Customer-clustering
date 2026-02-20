import { useState, useRef, useEffect } from 'react';
import { biQuery } from '../api/client';
import { Send, Loader2, Sparkles } from 'lucide-react';

const SUGGESTIONS = [
  "Which segment has the highest lifetime value?",
  "Which segment has the highest churn risk?",
  "What is the average basket of segment 7?",
  "Which product category drives the most revenue?",
  "Total number of customers?",
  "What is the cancellation rate?",
];

export default function BusinessIntelligence() {
  const [messages, setMessages] = useState([
    { role: 'bot', content: 'Hello. I am the CustomerIQ Business Intelligence assistant. I have direct access to your 11 segments and KPIs. Ask me anything about your customer base.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  const scrollToBottom = () => endRef.current?.scrollIntoView({ behavior: 'smooth' });

  useEffect(() => { scrollToBottom(); }, [messages]);

  const handleSend = async (text) => {
    if (!text.trim() || loading) return;
    const userMsg = text.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setLoading(true);

    try {
      const res = await biQuery(userMsg);
      setMessages(prev => [...prev, {
        role: 'bot',
        content: res.answer,
        suggestions: res.suggested_questions,
        data: res.data
      }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'bot', content: 'Sorry, I encountered an error connecting to the BI engine.', isError: true }]);
    } finally {
      setLoading(false);
    }
  };

  const parseMarkdown = (text) => {
    // Simple bold parsing for answers
    return text.split(/(\*\*.*?\*\*)/).map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={i}>{part.slice(2, -2)}</strong>;
      }
      return part;
    });
  };

  return (
    <div className="page-enter" style={{ height: 'calc(100vh - 64px)' }}>
      <div className="page-header" style={{ marginBottom: 16 }}>
        <h1 className="page-title">Business Intelligence</h1>
        <p className="page-subtitle">Natural-language querying over the segmentation model and KPIs.</p>
      </div>

      <div className="bi-chat">
        <div className="bi-messages">
          {messages.map((msg, i) => (
            <div key={i} className={`bi-bubble bi-bubble-${msg.role}`} style={{ display: 'flex', gap: 12 }}>
              {msg.role === 'bot' && (
                <div style={{ flexShrink: 0, marginTop: 2 }}>
                  <Sparkles size={18} color="var(--indigo)" />
                </div>
              )}
              <div style={{ flex: 1 }}>
                <div>{msg.role === 'bot' ? parseMarkdown(msg.content) : msg.content}</div>
                {msg.suggestions && (
                  <div className="bi-pills" style={{ marginTop: 12, marginBottom: 0 }}>
                    {msg.suggestions.map(s => (
                      <button key={s} className="bi-pill" onClick={() => handleSend(s)}>{s}</button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="bi-bubble bi-bubble-bot" style={{ display: 'flex', gap: 12 }}>
              <div style={{ flexShrink: 0, marginTop: 2 }}><Sparkles size={18} color="var(--indigo)" /></div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Loader2 size={16} className="spinner" style={{ margin: 0 }} />
                <span style={{ color: '#94a3b8' }}>Analysing data...</span>
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        {messages.length === 1 && (
          <div className="bi-pills">
            {SUGGESTIONS.map(s => (
              <button key={s} className="bi-pill" onClick={() => handleSend(s)}>{s}</button>
            ))}
          </div>
        )}

        <form className="bi-input-row" onSubmit={(e) => { e.preventDefault(); handleSend(input); }}>
          <input
            type="text"
            className="bi-input"
            placeholder="Ask about retention, revenue, or segments... (e.g. 'What is the average basket of segment 7?')"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
          />
          <button type="submit" className="btn btn-primary" disabled={loading || !input.trim()}>
            <Send size={18} />
          </button>
        </form>
      </div>
    </div>
  );
}

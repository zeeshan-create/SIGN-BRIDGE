import React from "react";
import { AlertCircle, RefreshCcw, Home, ShieldAlert } from "lucide-react";
import { motion } from "framer-motion";

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("[System Error]", error, errorInfo);
  }

  handleReset = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-[#F0FDF4] flex items-center justify-center p-6 font-inter">
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="max-w-xl w-full bg-white rounded-[3rem] p-12 shadow-2xl border border-red-50 relative overflow-hidden"
          >
            {/* Background Decorative Blurs */}
            <div className="absolute -top-24 -right-24 w-64 h-64 bg-red-50 rounded-full blur-[80px] pointer-events-none opacity-50" />
            <div className="absolute -bottom-24 -left-24 w-64 h-64 bg-emerald-50 rounded-full blur-[80px] pointer-events-none opacity-50" />

            <div className="relative z-10 text-center">
              <div className="w-20 h-20 bg-red-50 text-red-500 rounded-3xl flex items-center justify-center mx-auto mb-8 shadow-inner">
                <ShieldAlert size={40} />
              </div>
              
              <h1 className="text-3xl font-black text-[#052E16] tracking-tight mb-4 leading-tight">
                Neural Interface <br /> <span className="text-red-500 italic">Interrupted.</span>
              </h1>
              
              <p className="text-gray-500 font-medium text-lg leading-relaxed mb-10 tracking-tight">
                An unexpected sequence error occurred in the front-end rendering pipeline. We've isolated the failure to protect the underlying Neural Bridge.
              </p>

              <div className="bg-gray-50/50 rounded-2xl p-6 mb-10 border border-gray-100 text-left">
                <div className="flex items-center gap-3 mb-2">
                  <AlertCircle size={14} className="text-red-500" />
                  <span className="text-[10px] font-black uppercase tracking-widest text-red-500">Error Stack Trace</span>
                </div>
                <code className="text-[11px] font-mono text-gray-400 break-all leading-relaxed line-clamp-3">
                  {this.state.error?.message || "Internal Rendering Sequencer Failure"}
                </code>
              </div>

              <div className="flex flex-col sm:flex-row gap-4">
                <button 
                  onClick={this.handleReset}
                  className="flex-1 bg-[#052E16] text-white px-8 py-4 rounded-2xl text-[0.75rem] font-black uppercase tracking-widest hover:bg-[#16A34A] transition-all flex items-center justify-center gap-3 shadow-xl shadow-[#052E16]/20"
                >
                  <RefreshCcw size={18} /> Re-Initialize
                </button>
                <a 
                  href="/"
                  className="flex-1 bg-white text-[#052E16] border border-gray-100 px-8 py-4 rounded-2xl text-[0.75rem] font-black uppercase tracking-widest hover:bg-gray-50 transition-all flex items-center justify-center gap-3 shadow-sm"
                >
                  <Home size={18} /> Return Home
                </a>
              </div>
            </div>
          </motion.div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;

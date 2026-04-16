import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Leaf, CheckCircle2, Github, Linkedin, Mail, MapPin, Sparkles, ChevronDown } from 'lucide-react';

function Contact({ socket }) {
  const [formData, setFormData] = useState({ name: '', email: '', message: '' });
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if(socket) socket.emit('save_message', formData);
    setSubmitted(true);
    setFormData({ name: '', email: '', message: '' });
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.15, ease: [0.22, 1, 0.36, 1] }}
      className="min-h-screen bg-[var(--bg-primary)] font-inter pb-32 relative overflow-hidden transition-colors duration-300"
    >
      {/* Ambient Lighting Backgrounds */}
      <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-gradient-to-bl from-[#10b981]/10 to-transparent rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute top-1/2 left-0 w-[800px] h-[800px] bg-gradient-to-tr from-emerald-50/60 to-transparent rounded-full blur-[120px] pointer-events-none" />

      <div className="max-w-[1400px] mx-auto px-6 py-32 md:py-48 relative z-10">
        
        <header className="mb-24 text-center">
          <motion.div initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="inline-flex items-center gap-2 px-4 py-2 bg-[var(--bg-tertiary)] text-[#10b981] rounded-full text-[10px] font-black uppercase tracking-widest mb-8 border border-[var(--border-color)] shadow-sm">
             <Sparkles size={14} /> Inquiry Portal
          </motion.div>
          <motion.h1 initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.02 }} className="text-5xl md:text-7xl font-black tracking-tight text-[var(--text-primary)] mb-6">
            Connect with <span className="text-[#10b981]">SignBridge.</span>
          </motion.h1>
          <motion.p initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.04 }} className="text-xl text-[var(--text-secondary)] font-medium max-w-xl mx-auto leading-relaxed">
            Have a question, feedback, or a partnership inquiry? Reach out and we'll bridge the neural gap together.
          </motion.p>
        </header>

        <div className="max-w-[800px] mx-auto">
          {/* Form Section */}
          <div className="w-full">
        {submitted ? (
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-[var(--bg-secondary)] border border-[var(--border-color)] p-20 rounded-[4rem] text-center shadow-2xl relative overflow-hidden"
          >
             <div className="absolute inset-0 bg-gradient-to-b from-[#10b981]/5 to-transparent pointer-events-none" />
             <div className="w-32 h-32 bg-[#10b981]/10 text-[#10b981] rounded-[2.5rem] flex items-center justify-center mx-auto mb-10 shadow-xl border border-[#10b981]/20 relative z-10">
                <CheckCircle2 size={56} />
             </div>
             <h2 className="text-4xl font-black text-[var(--text-primary)] mb-4 tracking-tight relative z-10">Message Received.</h2>
             <p className="text-[var(--text-secondary)] mb-12 font-medium text-lg max-w-sm mx-auto relative z-10">
               Thank you for reaching out. A specialist from our core lab will evaluate your inquiry shortly.
             </p>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setSubmitted(false)}
                className="relative z-10 text-emerald-600 font-black uppercase tracking-[0.2em] text-[10px] hover:text-[#052E16] transition-colors border-b-2 border-emerald-200 pb-1"
              >
               Initialize New Dispatch
             </motion.button>
          </motion.div>
        ) : (
          <motion.form 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.06 }}
            onSubmit={handleSubmit} 
            className="space-y-10 bg-[var(--bg-secondary)] backdrop-blur-xl p-10 md:p-16 rounded-[4rem] border border-[var(--border-color)] shadow-2xl relative"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
              <div className="space-y-4">
                 <label className="text-[0.65rem] font-black uppercase tracking-[0.4em] text-[var(--text-tertiary)] ml-4">Full Identity</label>
                 <input 
                   required
                   type="text" 
                   className="w-full bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded-[2rem] p-6 text-sm font-black text-[var(--text-primary)] focus:ring-4 focus:ring-[#10b981]/20 focus:border-[#10b981] outline-none transition-all placeholder:text-[var(--text-tertiary)]"
                   placeholder="Your Name"
                   value={formData.name}
                   onChange={(e) => setFormData({...formData, name: e.target.value})}
                 />
              </div>
              <div className="space-y-4">
                 <label className="text-[0.65rem] font-black uppercase tracking-[0.4em] text-[var(--text-tertiary)] ml-4">Email Uplink</label>
                 <input 
                   required
                   type="email" 
                   className="w-full bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded-[2rem] p-6 text-sm font-black text-[var(--text-primary)] focus:ring-4 focus:ring-[#10b981]/20 focus:border-[#10b981] outline-none transition-all placeholder:text-[var(--text-tertiary)]"
                   placeholder="name@architecture.com"
                   value={formData.email}
                   onChange={(e) => setFormData({...formData, email: e.target.value})}
                 />
              </div>
            </div>

            <div className="space-y-4">
               <label className="text-[0.65rem] font-black uppercase tracking-[0.4em] text-[var(--text-tertiary)] ml-4">Message Transcription</label>
               <textarea 
                 required
                 rows={6}
                 className="w-full bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded-[2.5rem] p-8 text-sm font-black text-[var(--text-primary)] focus:ring-4 focus:ring-[#10b981]/20 focus:border-[#10b981] outline-none transition-all placeholder:text-[var(--text-tertiary)] resize-none"
                 placeholder="How can we assist your bridge experience?"
                 value={formData.message}
                 onChange={(e) => setFormData({...formData, message: e.target.value})}
               />
            </div>

            <motion.button
              type="submit"
              whileHover={{ scale: 1.02, y: -2, boxShadow: "0 0 30px rgba(16, 185, 129, 0.4)" }}
              whileTap={{ scale: 0.98 }}
              className="group relative bg-[#052E16] text-white px-16 py-6 rounded-[2rem] font-black text-[11px] uppercase tracking-[0.3em] shadow-xl shadow-[#052E16]/20 transition-all hover:bg-[#16A34A] hover:shadow-[#16A34A]/30 w-full flex items-center justify-center gap-4 overflow-hidden"
            >
              <span className="relative z-10 flex items-center gap-4">Transmit Protocol <Send size={16} /></span>
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
            </motion.button>
          </motion.form>
        )}
        </div>
        </div>

        {/* Info Grid with user links */}
        <div className="mt-32 pt-24 border-t border-[var(--border-color)] grid grid-cols-1 md:grid-cols-3 gap-12 text-center relative z-10">
           
           <motion.a whileHover={{ scale: 1.02, y: -5 }} whileTap={{ scale: 0.98 }} href="https://github.com/zeeshan-create" target="_blank" rel="noopener noreferrer" className="group bg-[var(--bg-secondary)] p-10 rounded-[2.5rem] border border-[var(--border-color)] shadow-sm hover:shadow-xl hover:border-[#10b981] transition-all duration-300 flex flex-col items-center">
               <div className="w-12 h-12 bg-[var(--bg-tertiary)] text-[var(--text-secondary)] rounded-xl flex items-center justify-center mb-6 group-hover:bg-[#10b981] group-hover:text-white transition-colors">
                 <Github size={20} />
               </div>
               <p className="text-[0.65rem] font-black uppercase tracking-[0.5em] text-[var(--text-tertiary)] mb-2">GitHub Repository</p>
               <p className="font-black text-lg text-[var(--text-primary)] tracking-tight">zeeshan-create</p>
            </motion.a>

            <motion.a whileHover={{ scale: 1.02, y: -5 }} whileTap={{ scale: 0.98 }} href="https://www.linkedin.com/in/zeeshan-hussain-32b9a2286/" target="_blank" rel="noopener noreferrer" className="group bg-[var(--bg-secondary)] p-10 rounded-[2.5rem] border border-[var(--border-color)] shadow-sm hover:shadow-xl hover:border-[#0A66C2] transition-all duration-300 flex flex-col items-center">
               <div className="w-12 h-12 bg-[var(--bg-tertiary)] text-[#0A66C2] rounded-xl flex items-center justify-center mb-6 group-hover:bg-[#0A66C2] group-hover:text-white transition-colors">
                 <Linkedin size={20} />
               </div>
               <p className="text-[0.65rem] font-black uppercase tracking-[0.5em] text-[var(--text-tertiary)] mb-2">Professional Network</p>
               <p className="font-black text-lg text-[var(--text-primary)] tracking-tight">Zeeshan Hussain</p>
            </motion.a>

            <motion.div whileHover={{ scale: 1.02, y: -5 }} whileTap={{ scale: 0.98 }} className="group bg-[#052E16] dark:bg-[#02140A] p-10 rounded-[2.5rem] border border-[#10b981]/20 shadow-2xl transition-all duration-300 flex flex-col items-center relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-[#16A34A] rounded-full blur-[50px] opacity-20 pointer-events-none" />
              <div className="w-12 h-12 bg-white/10 text-[#BBF7D0] rounded-xl border border-white/5 flex items-center justify-center mb-6 relative z-10">
                <Mail size={20} />
              </div>
              <p className="text-[0.65rem] font-black uppercase tracking-[0.5em] text-emerald-100/50 mb-2 relative z-10">Direct Uplink</p>
              <p className="font-black text-sm text-white tracking-tight relative z-10">mzeeshanhussain07@gmail.com</p>
           </motion.div>
           
        </div>
      </div>
    </motion.div>
  );
}

export default Contact;

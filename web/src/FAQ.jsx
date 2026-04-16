import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, ChevronDown, HelpCircle, Shield, Info, Camera, Zap, Send, CheckCircle2 } from 'lucide-react';

function FAQ({ socket }) {
  const [openIndex, setOpenIndex] = useState(null);
  const [formData, setFormData] = useState({ name: '', email: '', message: '' });
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if(socket) socket.emit('save_message', formData);
    setSubmitted(true);
    setFormData({ name: '', email: '', message: '' });
  };

  const faqs = [
    {
      q: "What does SignBridge do?",
      a: "SignBridge translates American Sign Language (ASL) gestures from video or webcam into readable text using AI. It bridges the gap between sign language and textual communication in real-time.",
      icon: <Zap size={20} className="text-emerald-500" />
    },
    {
      q: "How accurate is the translation?",
      a: "Accuracy depends on conditions.\n\n• Works best with clear lighting and visible hands\n• Struggles with fast or unclear gestures\n\nThis system performs well on trained signs but is a prototype and not yet human-equivalent.",
      icon: <HelpCircle size={20} className="text-blue-500" />
    },
    {
      q: "Does it support full ASL sentences?",
      a: "No. SignBridge currently detects individual signs or gestures, not full conversations or grammar-based ASL structures. We are working on expanding context understanding.",
      icon: <Info size={20} className="text-purple-500" />
    },
    {
      q: "What input can I use?",
      a: "You can upload a video file or use your webcam directly. For optimal results, ensure your hand is clearly visible and centered in the frame.",
      icon: <Camera size={20} className="text-amber-500" />
    },
    {
      q: "Why are some predictions incorrect?",
      a: "Common reasons include poor lighting, complex backgrounds, partial hand visibility, or limited training data for specific gestures. AI models require clear visual data to maintain high confidence.",
      icon: <Sparkles size={20} className="text-[#10b981]" />
    },
    {
      q: "Is my data stored or shared?",
      a: "SignBridge does not permanently store your data. Uploaded videos are used only for processing and are not shared with third parties or stored on our servers permanently.",
      icon: <Shield size={20} className="text-emerald-600" />
    },
    {
      q: "Can this replace a human sign language interpreter?",
      a: "No. SignBridge is a prototype AI tool and does not match the accuracy, speed, or contextual understanding of a human interpreter. It is meant to assist, not replace, professional services.",
      icon: <HelpCircle size={20} className="text-gray-400" />
    }
  ];

  return (
    <motion.div 
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.15, ease: [0.22, 1, 0.36, 1] }}
      className="min-h-screen bg-[var(--bg-primary)] font-inter pb-32 transition-colors duration-300"
    >
      {/* Hero Section */}
      <section className="bg-[var(--bg-secondary)] border-b border-[var(--border-color)] pt-32 pb-24 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-gradient-to-bl from-emerald-100/30 to-transparent rounded-full blur-[100px] pointer-events-none" />
        <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-gradient-to-tr from-emerald-50/50 to-transparent rounded-full blur-[100px] pointer-events-none" />

        <div className="max-w-[1200px] mx-auto px-6 text-center relative z-10">
          <motion.div initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-50 text-emerald-700 rounded-full text-[10px] font-black uppercase tracking-widest mb-8 border border-emerald-100">
            <HelpCircle size={14} /> Knowledge Base
          </motion.div>
          <motion.h1 initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.02 }} className="text-6xl md:text-7xl font-black tracking-tight text-[var(--text-primary)] mb-8 leading-[1.1]">
            Common Queries & <br /> <span className="text-[#10b981]">Deep Insights.</span>
          </motion.h1>
          <p className="text-xl text-gray-500 font-medium leading-relaxed max-w-2xl mx-auto">
            Everything you need to know about how SignBridge operates, privacy protocols, and technical limitations.
          </p>
        </div>
      </section>

      {/* FAQ and Contact Support Grid */}
      <section className="max-w-[1400px] mx-auto px-6 py-24">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 lg:gap-24">
          {/* FAQ Section */}
          <div className="space-y-8">
            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-3xl font-black text-[#052E16] mb-12"
            >
              Frequently Asked Questions
            </motion.h2>
            {faqs.map((faq, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.01 }}
                className={`bg-[var(--nav-bg)] backdrop-blur-md rounded-[2.5rem] border transition-all duration-300 ${
                  openIndex === index
                    ? 'border-emerald-200 shadow-2xl shadow-emerald-500/5 ring-1 ring-emerald-100'
                    : 'border-[var(--border-color)] shadow-sm hover:shadow-md'
                }`}
              >
                <button
                  onClick={() => setOpenIndex(openIndex === index ? null : index)}
                  className="w-full text-left p-10 flex items-center justify-between gap-6 outline-none"
                >
                  <div className="flex items-center gap-6">
                     <div className={`w-14 h-14 rounded-2xl flex items-center justify-center transition-colors ${openIndex === index ? 'bg-emerald-50 text-emerald-600' : 'bg-[var(--bg-tertiary)] text-[var(--text-secondary)] shadow-inner'}`}>
                        {faq.icon}
                     </div>
                     <span className={`text-xl font-black tracking-tight transition-colors ${openIndex === index ? 'text-[var(--text-primary)]' : 'text-[var(--text-secondary)]'}`}>
                       {faq.q}
                     </span>
                  </div>
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center transition-all ${openIndex === index ? 'bg-[#052E16] text-white rotate-180' : 'bg-gray-50 text-gray-400'}`}>
                     <ChevronDown size={20} />
                  </div>
                </button>

                <AnimatePresence>
                  {openIndex === index && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.12, ease: [0.22, 1, 0.36, 1] }}
                      className="overflow-hidden"
                    >
                      <div className="px-10 pb-12 pt-0 ml-[5rem] text-gray-500 font-medium leading-relaxed whitespace-pre-line border-t border-emerald-50/50 pt-10 mt-2">
                        {faq.a}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            ))}
          </div>

          {/* Contact Support Section */}
          <div className="space-y-8">
            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-3xl font-black text-[var(--text-primary)] mb-12"
            >
              Contact Support
            </motion.h2>

            <div className="bg-[var(--bg-secondary)] backdrop-blur-xl p-8 rounded-[3rem] border border-[var(--border-color)] shadow-2xl shadow-emerald-900/5 relative">
              {submitted ? (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="text-center py-8"
                >
                  <div className="w-20 h-20 bg-emerald-50 text-emerald-500 rounded-[2rem] flex items-center justify-center mx-auto mb-6 shadow-xl shadow-emerald-500/10 border border-emerald-100">
                    <CheckCircle2 size={40} />
                  </div>
                  <h3 className="text-2xl font-black text-[#052E16] mb-2 tracking-tight">Message Received</h3>
                  <p className="text-gray-500 mb-6 font-medium text-sm">
                    Thank you for reaching out. We'll get back to you soon.
                  </p>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setSubmitted(false)}
                    className="text-emerald-600 font-black uppercase tracking-[0.2em] text-[10px] hover:text-[#052E16] transition-colors border-b-2 border-emerald-200 pb-1"
                  >
                   Send Another Message
                  </motion.button>
                </motion.div>
              ) : (
                <motion.form
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 }}
                  onSubmit={handleSubmit}
                  className="space-y-6"
                >
                  <div className="space-y-4">
                    <label className="text-[0.65rem] font-black uppercase tracking-[0.4em] text-[var(--text-tertiary)] ml-2">Full Name</label>
                    <input
                      required
                      type="text"
                      className="w-full bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded-[2rem] p-5 text-sm font-black text-[var(--text-primary)] focus:ring-4 focus:ring-emerald-500/10 focus:border-emerald-300 outline-none transition-all placeholder:text-gray-300"
                      placeholder="Your Name"
                      value={formData.name}
                      onChange={(e) => setFormData({...formData, name: e.target.value})}
                    />
                  </div>
                  <div className="space-y-4">
                    <label className="text-[0.65rem] font-black uppercase tracking-[0.4em] text-[var(--text-tertiary)] ml-2">Email Address</label>
                    <input
                      required
                      type="email"
                      className="w-full bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded-[2rem] p-5 text-sm font-black text-[var(--text-primary)] focus:ring-4 focus:ring-emerald-500/10 focus:border-emerald-300 outline-none transition-all placeholder:text-gray-300"
                      placeholder="your@email.com"
                      value={formData.email}
                      onChange={(e) => setFormData({...formData, email: e.target.value})}
                    />
                  </div>

                  <div className="space-y-4">
                    <label className="text-[0.65rem] font-black uppercase tracking-[0.4em] text-[var(--text-tertiary)] ml-2">Message</label>
                    <textarea
                      required
                      rows={4}
                      className="w-full bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded-[2rem] p-5 text-sm font-black text-[var(--text-primary)] focus:ring-4 focus:ring-emerald-500/10 focus:border-emerald-300 outline-none transition-all placeholder:text-gray-300 resize-none"
                      placeholder="How can we help you?"
                      value={formData.message}
                      onChange={(e) => setFormData({...formData, message: e.target.value})}
                    />
                  </div>

                  <motion.button
                    type="submit"
                    whileHover={{ scale: 1.02, y: -1, boxShadow: "0 0 20px rgba(16, 185, 129, 0.3)" }}
                    whileTap={{ scale: 0.98 }}
                    className="group relative bg-[#052E16] text-white px-8 py-4 rounded-[2rem] font-black text-[11px] uppercase tracking-[0.3em] shadow-xl shadow-[#052E16]/20 transition-all hover:bg-[#16A34A] hover:shadow-[#16A34A]/30 w-full flex items-center justify-center gap-3 overflow-hidden"
                  >
                    <span className="relative z-10 flex items-center gap-3">Send Message <Send size={14} /></span>
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
                  </motion.button>
                </motion.form>
              )}
            </div>
          </div>
        </div>
      </section>


    </motion.div>
  );
}

export default FAQ;

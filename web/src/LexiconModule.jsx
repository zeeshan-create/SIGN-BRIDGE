import React, { useState, unstable_batchedUpdates } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Delete, Keyboard, Type, Zap, ArrowRight, CheckCircle2 } from 'lucide-react';

const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const VALID_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'.split('');

const ASL_DICTIONARY = {
  "how are you": "YOU HOW",
  "what is your name": "YOUR NAME WHAT",
  "nice to meet you": "NICE MEET YOU",
  "where are you from": "YOU FROM WHERE",
  "what time is it": "TIME WHAT",
  "good morning": "GOOD MORNING",
  "good night": "GOOD NIGHT",
  "thank you": "THANK YOU",
  "you are welcome": "WELCOME",
  "please": "PLEASE",
  "yes": "YES",
  "no": "NO",
  "i love you": "I LOVE YOU",
  "see you later": "SEE YOU LATER",
  "how much": "HOW MUCH",
  "how many": "HOW MANY"
};

export default function LexiconModule({ socket }) {
  const [englishInput, setEnglishInput] = useState('');
  const [pipelineState, setPipelineState] = useState('idle'); // idle, cleaning, mapping, ready
  const [mappedASL, setMappedASL] = useState('');
  
  const [input, setInput] = useState(''); // The string to animate
  const [activeLetter, setActiveLetter] = useState(null);
  const [animationIndex, setAnimationIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Directly sets and translates text
  const handleTranslate = () => {
    if (!englishInput.trim()) return;
    setPipelineState('cleaning');
    
    // Simulate pipeline delay
    setTimeout(() => {
      // Clean string
      let cleaned = englishInput.toLowerCase().replace(/[^\w\s]|_/g, "").replace(/\s+/g, " ").trim();
      setPipelineState('mapping');
      
      setTimeout(() => {
        let result = ASL_DICTIONARY[cleaned] || cleaned.toUpperCase();
        setMappedASL(result);
        setInput(result); // Feed to animation player
        setPipelineState('ready');
        
        // Show first letter ready
        setActiveLetter(result[0] || null);
      }, 600);
    }, 600);
  };

  const handleClear = () => {
    setEnglishInput('');
    setMappedASL('');
    setPipelineState('idle');
    setInput('');
    setActiveLetter(null);
    setAnimationIndex(0);
    setIsPlaying(false);
    if (socket) socket.emit('clear_sentence');
  };

  const handlePlaySequence = () => {
    if (input.length === 0) return;
    setIsPlaying(true);
    setAnimationIndex(0);
    
    let index = 0;
    const interval = setInterval(() => {
      if (index >= input.length) {
        clearInterval(interval);
        setIsPlaying(false);
        return;
      }
      const char = input[index].toUpperCase();
      unstable_batchedUpdates(() => {
        setActiveLetter(char);
        setAnimationIndex(index);
      });
      if (socket && char !== ' ') socket.emit('manual_input', char);
      index++;
    }, 800); // 800ms per sign to watch it clearly
  };

  return (
    <section className="py-24 px-10 bg-[var(--bg-primary)] transition-colors duration-300">
      <div className="max-w-[1400px] mx-auto grid lg:grid-cols-2 gap-16 xl:gap-20 items-stretch">
        
        {/* Left Side: Visualizer */}
        <div className="space-y-12 flex flex-col justify-center">
          <header className="space-y-4">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-[var(--bg-tertiary)] border border-[var(--border-color)] text-[#10b981] rounded-full text-[10px] font-black uppercase tracking-widest shadow-sm">
              <Zap size={14} /> Translation Pipeline v1
            </div>
            <h2 className="text-5xl md:text-6xl font-black text-[var(--text-primary)] tracking-tight">
              English to ASL<br/>
              <span className="text-[#10b981]">Visualizer.</span>
            </h2>
            <p className="text-xl text-[var(--text-secondary)] font-medium max-w-lg leading-relaxed">
              Convert English text into structured ASL sign representations using an AI-powered translation pipeline.
            </p>
          </header>

          <div className="relative group perspective-1000">
            <div className="absolute -inset-4 bg-gradient-to-r from-[#10b981]/20 to-teal-500/10 rounded-[3rem] blur-2xl opacity-20 group-hover:opacity-40 transition-opacity duration-1000" />
             <div className="relative bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-[2.5rem] p-8 pb-20 shadow-2xl overflow-hidden aspect-video flex flex-col items-center justify-center transform-gpu transition-all">
              
               <AnimatePresence mode="popLayout">
                {activeLetter ? (
                   <motion.div
                     key={activeLetter}
                     initial={{ opacity: 0, scale: 0.9, y: 10 }}
                     animate={{ opacity: 1, scale: 1, y: 0 }}
                     exit={{ opacity: 0, scale: 1.1, y: -10 }}
                     transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
                     className="flex flex-col items-center gap-6"
                     style={{ willChange: 'transform, opacity' }}
                    >
                       <div className="w-52 h-52 bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded-[2rem] flex items-center justify-center overflow-hidden shadow-2xl relative">
                         {activeLetter === ' ' ? (
                            <div className="text-[#10b981] text-3xl font-black tracking-widest">SPACE</div>
                         ) : VALID_CHARS.includes(activeLetter) ? (
                            <img src={`/asl/${activeLetter}.jpg`} alt={`ASL sign for ${activeLetter}`} className="w-full h-full object-cover rounded-[2rem]" />
                         ) : (
                            <div className="text-[var(--text-tertiary)] text-3xl font-black">?</div>
                         )}
                         {/* Scanline overlay */}
                         <div className="absolute inset-0 bg-[linear-gradient(transparent_50%,rgba(16,185,129,0.05)_50%)] bg-[length:100%_4px] pointer-events-none opacity-50" />
                       </div>
                       <div className="text-center">
                         <p className="text-xs font-black uppercase tracking-[0.4em] text-[#10b981] mt-2">Sign Mapping Active: {activeLetter}</p>
                       </div>
                      </motion.div>
                ) : (
                   <motion.div
                     initial={{ opacity: 0, scale: 0.95 }}
                     animate={{ opacity: 1, scale: 1 }}
                     transition={{ duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
                     className="text-center space-y-4"
                     style={{ willChange: 'transform, opacity' }}
                   >
                    <div className="w-20 h-20 bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded-3xl flex items-center justify-center mx-auto text-[#10b981]">
                      <Type size={40} />
                    </div>
                    <p className="text-[var(--text-tertiary)] font-bold uppercase tracking-widest text-xs">
                      Enter English text to begin translation
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>

               {/* Translation Box */}
               {mappedASL && (
                 <motion.div 
                   initial={{ opacity: 0, y: 20 }}
                   animate={{ opacity: 1, y: 0 }}
                   className="absolute bottom-6 left-6 right-6 bg-[var(--nav-bg)] border border-[var(--border-color)] text-[var(--text-primary)] p-5 rounded-xl flex items-center justify-between shadow-xl gap-3 backdrop-blur-xl"
                 >
                   <div className="flex items-center gap-4 overflow-hidden flex-1">
                     <span className="text-[#10b981] font-black text-[10px] uppercase tracking-widest min-w-[70px]">Output:</span>
                     <p className="text-lg font-bold tracking-[0.2em] truncate flex flex-wrap gap-[2px]">
                       {mappedASL.split('').map((char, i) => (
                         <span 
                           key={i} 
                           className={`inline-block transition-all ${i === animationIndex && isPlaying ? 'text-[#10b981] scale-125 font-black bg-[#10b981]/10 px-1 rounded' : char === ' ' ? 'w-4' : ''}`}
                         >
                           {char}
                         </span>
                       ))}
                     </p>
                   </div>
                   <div className="flex gap-2">
                     <button 
                       onClick={handlePlaySequence}
                       disabled={isPlaying}
                       className="p-2 hover:bg-[#10b981]/20 rounded-lg transition-colors text-[#10b981] disabled:opacity-30 disabled:cursor-not-allowed"
                     >
                       <Zap size={20} />
                     </button>
                   </div>
                 </motion.div>
               )}
            </div>
          </div>
        </div>

        {/* Right Side: Pipeline Logic UI */}
        <div className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-[3rem] p-10 flex flex-col justify-between shadow-sm transition-colors duration-300">
           <div>
              <div className="flex items-center justify-between border-b border-[var(--border-color)] pb-8 mb-8">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded-xl flex items-center justify-center shadow-sm text-[var(--text-primary)]">
                    <Keyboard size={24} />
                  </div>
                  <div>
                    <h3 className="font-black text-[var(--text-primary)] uppercase tracking-widest text-[0.7rem]">Structure Mapping Engine</h3>
                    <p className="text-xs text-[var(--text-tertiary)] font-bold mt-1">ASL Grammar Restructuring module</p>
                  </div>
                </div>
              </div>

              {/* Text Input */}
              <div className="space-y-4">
                 <label className="text-xs font-black text-[var(--text-secondary)] uppercase tracking-widest">Input English Sentence</label>
                 <textarea
                   value={englishInput}
                   onChange={e => setEnglishInput(e.target.value)}
                   disabled={pipelineState === 'cleaning' || pipelineState === 'mapping'}
                   placeholder="e.g. How are you?"
                   className="w-full h-32 bg-[var(--bg-primary)] border border-[var(--border-color)] rounded-2xl p-6 text-[var(--text-primary)] font-medium text-lg resize-none focus:outline-none focus:border-[#10b981] transition-colors disabled:opacity-50"
                 />
                 
                 <div className="flex gap-4">
                    <button 
                      onClick={handleTranslate}
                      disabled={pipelineState !== 'idle' && pipelineState !== 'ready'}
                      className="flex-1 py-4 bg-[#10b981] text-white rounded-2xl font-black uppercase tracking-widest text-[11px] hover:bg-emerald-600 transition-colors disabled:opacity-50"
                    >
                      Process & Translate
                    </button>
                    <button 
                      onClick={handleClear}
                      className="px-6 py-4 bg-[var(--bg-tertiary)] text-red-400 border border-[var(--border-color)] hover:border-red-500/30 hover:bg-red-500/10 rounded-2xl font-black uppercase tracking-widest text-[11px] transition-colors"
                    >
                      Clear
                    </button>
                 </div>
              </div>

              {/* Pipeline Visualizer */}
              <div className="mt-12 space-y-6">
                 <h4 className="text-[10px] font-black text-[var(--text-tertiary)] uppercase tracking-[0.2em] border-b border-[var(--border-color)] pb-3">Translation Pipeline Status</h4>
                 
                 <div className="space-y-4">
                    {/* Step 1: Cleaning */}
                    <div className={`p-4 rounded-xl border flex items-center justify-between transition-all ${pipelineState === 'cleaning' ? 'bg-[#10b981]/10 border-[#10b981] text-[#10b981]' : pipelineState === 'mapping' || pipelineState === 'ready' ? 'bg-[var(--bg-primary)] border-[var(--border-color)] text-[var(--text-primary)]' : 'bg-[var(--bg-tertiary)] border-[var(--border-color)] text-[var(--text-tertiary)] opacity-50'}`}>
                       <div className="flex items-center gap-3">
                         <div className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-black ${pipelineState === 'cleaning' || pipelineState === 'mapping' || pipelineState === 'ready' ? 'bg-current text-white dark:text-[#052E16]' : 'bg-[var(--border-color)]'}`}>1</div>
                         <span className="font-bold text-sm tracking-wide">Text Cleaning & Normalization</span>
                       </div>
                       {(pipelineState === 'mapping' || pipelineState === 'ready') && <CheckCircle2 size={20} className="text-[#10b981]" />}
                    </div>

                    {/* Step 2: Grammar Restructuring */}
                    <div className={`p-4 rounded-xl border flex items-center justify-between transition-all ${pipelineState === 'mapping' ? 'bg-[#10b981]/10 border-[#10b981] text-[#10b981]' : pipelineState === 'ready' ? 'bg-[var(--bg-primary)] border-[var(--border-color)] text-[var(--text-primary)]' : 'bg-[var(--bg-tertiary)] border-[var(--border-color)] text-[var(--text-tertiary)] opacity-50'}`}>
                       <div className="flex items-center gap-3">
                         <div className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-black ${pipelineState === 'mapping' || pipelineState === 'ready' ? 'bg-current text-white dark:text-[#052E16]' : 'bg-[var(--border-color)]'}`}>2</div>
                         <span className="font-bold text-sm tracking-wide">ASL Grammar Restructuring</span>
                       </div>
                       {pipelineState === 'ready' && <CheckCircle2 size={20} className="text-[#10b981]" />}
                    </div>

                    {/* Step 3: Sign Representation Output */}
                    <div className={`p-4 rounded-xl border flex items-center justify-between transition-all ${pipelineState === 'ready' ? 'bg-[#10b981]/10 border-[#10b981] text-[#10b981] shadow-lg' : 'bg-[var(--bg-tertiary)] border-[var(--border-color)] text-[var(--text-tertiary)] opacity-50'}`}>
                       <div className="flex items-center gap-3">
                         <div className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-black ${pipelineState === 'ready' ? 'bg-current text-white dark:text-[#052E16]' : 'bg-[var(--border-color)]'}`}>3</div>
                         <span className="font-bold text-sm tracking-wide">Sign Representation Mapping</span>
                       </div>
                       {pipelineState === 'ready' && <Zap size={20} className="text-[#10b981] animate-pulse" />}
                    </div>
                 </div>
              </div>
           </div>

           {/* Final comparison display */}
           <AnimatePresence>
              {pipelineState === 'ready' && (
                 <motion.div 
                   initial={{ opacity: 0, y: 10 }}
                   animate={{ opacity: 1, y: 0 }}
                   className="mt-8 p-5 bg-[var(--bg-primary)] border border-dashed border-[#10b981]/50 rounded-2xl flex flex-col gap-3"
                 >
                    <div className="flex justify-between items-center text-xs font-black uppercase tracking-widest text-[var(--text-tertiary)]">
                       <span>English</span>
                       <ArrowRight size={14} className="text-[var(--border-color)]" />
                       <span className="text-[#10b981]">ASL Structure</span>
                    </div>
                    <div className="flex justify-between items-center font-bold">
                       <span className="text-[var(--text-primary)] truncate max-w-[45%]">{englishInput}</span>
                       <span className="text-[#10b981] truncate max-w-[45%] text-right">{mappedASL}</span>
                    </div>
                 </motion.div>
              )}
           </AnimatePresence>
        </div>

      </div>
    </section>
  );
}

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  LayoutGrid, Search, ChevronRight, BookOpen, Zap, Clock, 
  TrendingUp, Target, Star, Lock, CheckCircle, ArrowRight,
  Waves, Hash, MessageSquare, Users, Coffee, MapPin, Plus,
  Eye, Activity, Database
} from 'lucide-react';

const categoryIcons = {
  'Greetings': Waves,
  'Numbers 1-50': Hash,
  'Common Phrases': MessageSquare,
  'Family & People': Users,
  'Food & Drink': Coffee,
  'Places': MapPin,
};

const sampleVocab = {
  'Greetings': [
    { word: 'Hello', status: 'mastered' },
    { word: 'Good morning', status: 'mastered' },
    { word: 'Goodbye', status: 'mastered' },
    { word: 'Thank you', status: 'learning' },
    { word: 'Please', status: 'mastered' },
    { word: 'Sorry', status: 'learning' },
  ],
  'Numbers 1-50': [
    { word: 'One', status: 'mastered' },
    { word: 'Two', status: 'mastered' },
    { word: 'Three', status: 'learning' },
    { word: 'Four', status: 'mastered' },
    { word: 'Five', status: 'learning' },
  ],
  'Common Phrases': [
    { word: 'How are you?', status: 'mastered' },
    { word: 'Nice to meet you', status: 'learning' },
    { word: 'What time is it?', status: 'new' },
    { word: 'I understand', status: 'mastered' },
  ],
  'Family & People': [
    { word: 'Mother', status: 'mastered' },
    { word: 'Father', status: 'mastered' },
    { word: 'Sister', status: 'learning' },
    { word: 'Brother', status: 'new' },
  ],
  'Food & Drink': [
    { word: 'Water', status: 'mastered' },
    { word: 'Food', status: 'learning' },
    { word: 'Apple', status: 'new' },
    { word: 'Bread', status: 'new' },
  ],
  'Places': [
    { word: 'Home', status: 'mastered' },
    { word: 'School', status: 'learning' },
    { word: 'Store', status: 'new' },
    { word: 'Hospital', status: 'new' },
  ],
};

const initialCategories = [
  { 
    id: 'greetings',
    name: 'Greetings', 
    icon: Waves, 
    progress: 92, 
    count: '48/52 Signs',
    description: 'Essential introductions and polite expressions',
    total: 52,
    mastered: 48
  },
  { 
    id: 'numbers',
    name: 'Numbers 1-50', 
    icon: Hash, 
    progress: 85, 
    count: '170/200',
    description: 'Counting and numerical expressions',
    total: 200,
    mastered: 170
  },
  { 
    id: 'phrases',
    name: 'Common Phrases', 
    icon: MessageSquare, 
    progress: 67, 
    count: '134/200',
    description: 'Frequently used conversational phrases',
    total: 200,
    mastered: 134
  },
  { 
    id: 'family',
    name: 'Family & People', 
    icon: Users, 
    progress: 74, 
    count: '74/100',
    description: 'Family members and people descriptions',
    total: 100,
    mastered: 74
  },
  { 
    id: 'food',
    name: 'Food & Drink', 
    icon: Coffee, 
    progress: 55, 
    count: '110/200',
    description: 'Food items, drinks, and dining vocabulary',
    total: 200,
    mastered: 110
  },
  { 
    id: 'places',
    name: 'Places', 
    icon: MapPin, 
    progress: 41, 
    count: '41/100',
    description: 'Locations, buildings, and directions',
    total: 100,
    mastered: 41
  },
];

const neuralTips = [
  { icon: Zap, text: 'Neural indexing is deploying detected signs to your profile' },
  { icon: Activity, text: 'Real-time vocabulary expansion active' },
  { icon: Database, text: 'Machine learning models adapting to your signing style' },
];

export default function CategoriesView() {
  const [categories, setCategories] = useState(initialCategories);
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filter, setFilter] = useState('all');

  const filteredCategories = categories.filter(cat => 
    cat.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getStatusColor = (status) => {
    switch(status) {
      case 'mastered': return 'bg-emerald-100 text-emerald-700 border-emerald-200';
      case 'learning': return 'bg-amber-100 text-amber-700 border-amber-200';
      case 'new': return 'bg-slate-100 text-slate-500 border-slate-200';
      default: return 'bg-gray-100 text-gray-500';
    }
  };

  const getStatusIcon = (status) => {
    switch(status) {
      case 'mastered': return <CheckCircle size={14} />;
      case 'learning': return <Clock size={14} />;
      case 'new': return <Plus size={14} />;
      default: return null;
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="min-h-screen bg-[#F0FDF4] p-8 pt-24"
    >
      <div className="max-w-[1600px] mx-auto">
        {/* Header */}
        <div className="flex justify-between items-end mb-16">
          <div>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-[#16A34A]/10 rounded-2xl flex items-center justify-center text-[#16A34A]">
                <LayoutGrid size={24} />
              </div>
              <span className="text-[10px] font-black text-[#16A34A] uppercase tracking-[0.4em]">Module Active</span>
            </div>
            <h1 className="text-4xl md:text-5xl font-black text-[#052E16] tracking-tight mb-2">Mastery Categories</h1>
            <p className="text-[#16A34A]/60 text-sm font-medium">Organize, track, and expand your ASL vocabulary</p>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="relative">
              <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" />
              <input 
                type="text"
                placeholder="Search categories..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 pr-6 py-4 bg-white border border-[#BBF7D0]/30 rounded-[1.5rem] text-sm font-medium text-[#052E16] placeholder:text-gray-300 focus:outline-none focus:border-[#16A34A]/50 w-64 shadow-sm"
              />
            </div>
          </div>
        </div>

        {/* Neural Status Bar */}
        <div className="bg-[#052E16] rounded-[2rem] p-6 mb-12 flex items-center justify-between overflow-hidden relative">
          <div className="absolute top-0 right-0 w-48 h-full bg-gradient-to-l from-[#16A34A]/10 to-transparent pointer-events-none" />
          <div className="flex items-center gap-8 relative z-10">
            {neuralTips.map((tip, i) => (
              <div key={i} className="flex items-center gap-3 text-emerald-400/80">
                <tip.icon size={16} className="animate-pulse" />
                <span className="text-[10px] font-medium uppercase tracking-wider">{tip.text}</span>
              </div>
            ))}
          </div>
          <div className="flex items-center gap-2 bg-emerald-500/20 px-4 py-2 rounded-full">
            <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
            <span className="text-emerald-400 text-[10px] font-black uppercase tracking-widest">Neural Active</span>
          </div>
        </div>

        <AnimatePresence mode="wait">
          {!selectedCategory ? (
            <motion.div 
              key="grid"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
            >
              {filteredCategories.map((cat, i) => (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.01 }}
                  key={cat.id}
                  onClick={() => setSelectedCategory(cat)}
                  className="bg-white border border-[#BBF7D0]/30 rounded-[2.5rem] p-8 shadow-[0_20px_40px_-15px_rgba(22,163,74,0.05)] hover:shadow-[0_40px_80px_-20px_rgba(22,163,74,0.1)] transition-all group cursor-pointer hover:-translate-y-1"
                >
                  <div className="flex justify-between items-start mb-8">
                    <div className="w-14 h-14 bg-[#F0FDF4] rounded-2xl flex items-center justify-center text-[#16A34A] group-hover:bg-[#16A34A] group-hover:text-white transition-all shadow-inner">
                      <cat.icon size={26} />
                    </div>
                    <ChevronRight size={18} className="text-[#BBF7D0] group-hover:text-[#16A34A] group-hover:translate-x-1 transition-all" />
                  </div>
                  
                  <h4 className="font-black text-[#052E16] text-lg mb-2 tracking-tight">{cat.name}</h4>
                  <p className="text-xs text-gray-400 font-medium mb-6">{cat.description}</p>
                  
                  <div className="flex justify-between items-end mb-4">
                    <span className="text-[10px] font-black text-gray-300 uppercase tracking-[0.2em]">{cat.count}</span>
                    <span className="text-sm font-black text-[#16A34A]">{cat.progress}%</span>
                  </div>
                  
                  <div className="h-2.5 bg-[#F0FDF4] rounded-full overflow-hidden border border-[#BBF7D0]/20">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${cat.progress}%` }}
                      transition={{ duration: 0.3, delay: 0.02 + (i * 0.02) }}
                      className="h-full bg-gradient-to-r from-[#16A34A] to-[#4ADE80] rounded-full"
                    />
                  </div>

                  <div className="flex items-center gap-4 mt-8 pt-6 border-t border-[#F0FDF4]">
                    <div className="flex -space-x-2">
                      {[1,2,3].map((_, j) => (
                        <div key={j} className="w-8 h-8 rounded-full bg-emerald-100 border-2 border-white flex items-center justify-center text-[10px] font-black text-emerald-600">
                          {String.fromCharCode(65 + j)}
                        </div>
                      ))}
                    </div>
                    <span className="text-[10px] text-gray-400 font-medium">+24 learners today</span>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          ) : (
            <motion.div 
              key="detail"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="bg-white border border-[#BBF7D0]/30 rounded-[3rem] p-10 shadow-xl"
            >
              {/* Detail Header */}
              <div className="flex items-center justify-between mb-10 pb-8 border-b border-[#F0FDF4]">
                <div className="flex items-center gap-6">
                  <button 
                    onClick={() => setSelectedCategory(null)}
                    className="w-12 h-12 bg-[#F0FDF4] rounded-2xl flex items-center justify-center text-[#16A34A] hover:bg-[#16A34A] hover:text-white transition-all"
                  >
                    <ArrowRight size={20} className="rotate-180" />
                  </button>
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <selectedCategory.icon size={24} className="text-[#16A34A]" />
                      <h2 className="text-3xl font-black text-[#052E16]">{selectedCategory.name}</h2>
                    </div>
                    <p className="text-sm text-gray-400 font-medium">{selectedCategory.description}</p>
                  </div>
                </div>
                
                <div className="flex items-center gap-8">
                  <div className="text-center">
                    <p className="text-[10px] font-black text-gray-300 uppercase tracking-[0.3em] mb-1">Progress</p>
                    <p className="text-3xl font-black text-[#16A34A]">{selectedCategory.progress}%</p>
                  </div>
                  <div className="text-center">
                    <p className="text-[10px] font-black text-gray-300 uppercase tracking-[0.3em] mb-1">Mastered</p>
                    <p className="text-3xl font-black text-[#052E16]">{selectedCategory.mastered}<span className="text-gray-300 text-lg">/{selectedCategory.total}</span></p>
                  </div>
                </div>
              </div>

              {/* Filter Tabs */}
              <div className="flex gap-4 mb-8">
                {['all', 'mastered', 'learning', 'new'].map((f) => (
                  <button
                    key={f}
                    onClick={() => setFilter(f)}
                    className={`px-6 py-3 rounded-[1.5rem] text-[10px] font-black uppercase tracking-widest transition-all ${
                      filter === f 
                        ? 'bg-[#16A34A] text-white shadow-lg shadow-[#16A34A]/20' 
                        : 'bg-[#F0FDF4] text-gray-500 hover:bg-[#BBF7D0] hover:text-[#16A34A]'
                    }`}
                  >
                    {f === 'all' ? 'All Words' : f.charAt(0).toUpperCase() + f.slice(1)}
                  </button>
                ))}
              </div>

              {/* Vocabulary List */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {(sampleVocab[selectedCategory.name] || [])
                  .filter(word => filter === 'all' || word.status === filter)
                  .map((word, i) => (
                    <motion.div 
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.006 }}
                      key={i}
                      className={`flex items-center justify-between p-6 rounded-[1.5rem] border transition-all cursor-pointer hover:shadow-md ${getStatusColor(word.status)}`}
                    >
                      <div className="flex items-center gap-4">
                        {getStatusIcon(word.status)}
                        <span className="font-black text-[#052E16] uppercase tracking-wide">{word.word}</span>
                      </div>
                      <ChevronRight size={16} className="text-gray-400" />
                    </motion.div>
                  ))}
              </div>

              {/* Add New Button */}
              <motion.button 
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="w-full mt-8 py-6 bg-[#F0FDF4] border-2 border-dashed border-[#BBF7D0]/50 rounded-[1.5rem] text-[#16A34A] font-black text-sm uppercase tracking-widest hover:bg-[#16A34A] hover:text-white hover:border-[#16A34A] transition-all flex items-center justify-center gap-3"
              >
                <Plus size={20} /> Add Custom Sign
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}
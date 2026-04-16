import React from 'react';
import { LayoutDashboard, History, LayoutGrid, Users, Settings, LogOut, Leaf } from 'lucide-react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

function Sidebar({ activeTab, setActiveTab, isOpen, onClose }) {
  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'interpreter', label: 'Sign interpreter', icon: History },
    { id: 'categories', label: 'Categories', icon: LayoutGrid },
    { id: 'community', label: 'Community', icon: Users },
  ];

  const handleTabClick = (id) => {
    setActiveTab(id);
    if (onClose) onClose();
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
  };

  const sidebarContent = (
    <aside className="w-64 bg-[var(--nav-bg)] backdrop-blur-2xl border-r border-[var(--border-color)] flex flex-col h-screen h-[100dvh] shadow-2xl transition-colors duration-300">
      {/* Brand Logo Section */}
      <div className="p-8 flex flex-col items-center gap-4">
         <Link to="/" className="w-full hover:scale-105 transition-transform duration-500">
            <div className="aspect-square w-28 mx-auto bg-white dark:bg-white/90 p-3 rounded-full shadow-sm border border-[var(--border-color)] flex items-center justify-center overflow-hidden">
              <img src="/logo.png" className="w-full h-full object-contain" alt="SignBridge" />
            </div>
         </Link>
         <div className="w-full h-px bg-gradient-to-r from-transparent via-[#16A34A]/10 to-transparent" />
      </div>

      {/* Nav */}
      <nav className="flex-1 px-4 space-y-3 mt-12 overflow-y-auto scrollbar-hide">
          {menuItems.map((item) => (
            <button
              key={item.id}
              onClick={() => handleTabClick(item.id)}
             className={`w-full flex flex-col items-center py-5 rounded-full transition-all duration-300 group ${
               activeTab === item.id 
                 ? 'bg-[#16A34A] text-white shadow-lg shadow-[#16A34A]/20' 
                 : 'text-[var(--text-secondary)] hover:text-[#16A34A] hover:bg-[var(--bg-tertiary)]'
             }`}
           >
             <item.icon size={22} className={`mb-2 transition-transform duration-300 group-hover:scale-110 ${activeTab === item.id ? 'text-white' : 'text-[#16A34A]/60 group-hover:text-[#16A34A]'}`} />
             <span className="text-[0.65rem] font-black uppercase tracking-[0.2em]">{item.label}</span>
           </button>
         ))}
      </nav>

      {/* Footer Nav */}
      <div className="p-4 space-y-2 mb-4">
          <button 
            onClick={() => handleTabClick('settings')}
           className={`w-full flex flex-col items-center py-5 rounded-full transition-all ${
             activeTab === 'settings' ? 'bg-[#16A34A] text-white shadow-lg shadow-[#16A34A]/20' : 'text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)]'
           }`}
         >
           <Settings size={22} className={`mb-2 ${activeTab === 'settings' ? 'text-white' : 'text-[#16A34A]/60'}`} />
           <span className="text-[0.65rem] font-black uppercase tracking-[0.2em]">Settings</span>
         </button>
      </div>
    </aside>
  );

  return (
    <>
      {/* Desktop Sidebar */}
      <div className="hidden lg:block fixed left-0 top-0 z-50">
        {sidebarContent}
      </div>

      {/* Mobile Sidebar Overlay */}
      <AnimatePresence>
        {isOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={onClose}
              className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[60] lg:hidden"
            />
            <motion.div
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed left-0 top-0 h-screen z-[70] lg:hidden"
            >
              {sidebarContent}
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}

export default Sidebar;

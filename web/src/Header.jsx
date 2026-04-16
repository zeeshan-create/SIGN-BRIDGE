import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from './AuthContext';
import { useTheme } from './ThemeContext';
import { User, Shield, Sun, Moon } from 'lucide-react';

const MotionLink = motion(Link);

function Header() {
  const location = useLocation();
  const path = location.pathname;
  const { user } = useAuth();
  const { isDark, toggleTheme } = useTheme();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleLinkClick = () => {
    setMobileMenuOpen(false);
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
  };

  const navItems = [
    { path: "/", label: "Home" },
    { path: "/dashboard", label: "Dashboard" },
    { path: "/converter", label: "Converter" },
    { path: "/about", label: "About" },
    { path: "/faq", label: "FAQ" },
    { path: "/contact", label: "Contact" },
  ];

  if (path.startsWith('/dashboard')) return null;

  return (
    <>
      <motion.nav
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.18, ease: [0.25, 1, 0.5, 1] }}
        className="w-full h-16 bg-black/95 backdrop-blur-md border-b border-white/10 flex items-center justify-between px-6 md:px-8 sticky top-0 z-[100] will-change-transform transform translate-z-0 contain layout paint style"
      >
        <div className="flex items-center gap-6 md:gap-12">
            <Link to="/" className="flex items-center gap-3 group" onClick={() => setMobileMenuOpen(false)}>
              <div className="w-10 h-10 md:w-12 md:h-12 rounded-full bg-white dark:bg-black/20 flex items-center justify-center border border-white/10 shadow-sm overflow-hidden p-1">
                <img src="/logo.png" className="w-full h-full object-contain logo-invert" alt="SignBridge" />
              </div>
               <span className="font-bold text-lg md:text-xl tracking-tight text-white">SignBridge</span>
            </Link>

           <div className="hidden md:flex items-center gap-1">
             {navItems.map((item, i) => (
               <MotionLink
                 key={item.path}
                 to={item.path}
                 onClick={handleLinkClick}
                 initial={{ opacity: 0, y: -8 }}
                 animate={{ 
                   opacity: 1, 
                   y: 0, 
                   scale: path === item.path ? 1.05 : 1,
                   color: path === item.path ? "#ffffff" : undefined,
                 }}
                 transition={{ delay: 0.01 + (i * 0.012), type: "spring", stiffness: 500 }}
                 whileHover={{ y: -2, scale: 1.08, color: "#ffffff" }}
                 whileTap={{ scale: 0.95 }}
                  className={`relative py-1.5 px-5 text-base transition-colors duration-200 ${
                   path === item.path 
                     ? 'text-white font-medium' 
                     : 'text-white/40 font-light hover:text-white/90'
                 }`}
               >
                 {item.label}
                 {path === item.path && (
                   <motion.div 
                     layoutId="nav-glow"
                     className="absolute inset-0 bg-white/5 rounded-lg -z-10"
                     transition={{ type: "spring", stiffness: 350, damping: 30 }}
                   />
                 )}
               </MotionLink>
             ))}
           </div>
        </div>
        
        <div className="flex items-center gap-2 md:gap-3">
          {user?.role === 'admin' && (
            <MotionLink
              to="/admin"
              onClick={handleLinkClick}
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.12, type: "spring", stiffness: 500 }}
              whileHover={{ y: -1 }}
              whileTap={{ scale: 0.96 }}
               className={`hidden sm:block py-1.5 px-4 text-sm font-light tracking-[0.05em] transition-colors duration-200 ${
                path === '/admin' ? 'text-white' : 'text-white/35 hover:text-white/80'
              }`}
            >
              Admin
            </MotionLink>
          )}
          
          <button
            onClick={toggleTheme}
            className="p-2 text-white/70 hover:text-white hover:bg-white/10 rounded-full transition-all active:scale-90"
            aria-label="Toggle Theme"
          >
            {isDark ? <Sun size={20} /> : <Moon size={20} />}
          </button>

          {user ? (
            <MotionLink
              to="/profile"
              onClick={handleLinkClick}
              whileHover={{ opacity: 0.8 }}
              whileTap={{ scale: 0.97 }}
              className="flex items-center gap-2"
            >
              <div className="w-7 h-7 md:w-8 md:h-8 rounded-full bg-[#10b981] flex items-center justify-center text-white text-[10px] md:text-xs font-bold shadow-lg shadow-[#10b981]/20">
                 {user.name.charAt(0)}
              </div>
            </MotionLink>
          ) : (
            <MotionLink
              to="/login"
              onClick={handleLinkClick}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.96 }}
               className="bg-white/10 text-white px-4 md:px-5 py-2 rounded-lg text-sm font-medium hover:bg-white/15 transition-colors duration-200"
            >
              Sign In
            </MotionLink>
          )}

          <button 
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 text-white/70 hover:text-white transition-colors"
          >
            {mobileMenuOpen ? <Shield size={24} /> : (
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="3" y1="12" x2="21" y2="12"></line>
                <line x1="3" y1="6" x2="21" y2="6"></line>
                <line x1="3" y1="18" x2="21" y2="18"></line>
              </svg>
            )}
          </button>
        </div>
      </motion.nav>

      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="fixed top-16 left-0 w-full bg-[var(--bg-secondary)] backdrop-blur-3xl z-50 overflow-hidden md:hidden border-b border-[var(--border-color)]"
          >
            <div className="flex flex-col p-8 gap-2">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  onClick={handleLinkClick}
                  className={`text-2xl font-black uppercase tracking-[0.2em] py-5 border-b border-[var(--border-color)] last:border-0 transition-all active:scale-95 ${
                    path === item.path ? 'text-[#16A34A]' : 'text-[var(--text-tertiary)] hover:text-[#16A34A]'
                  }`}
                >
                  {item.label}
                </Link>
              ))}
              {user?.role === 'admin' && (
                <Link
                  to="/admin"
                  onClick={handleLinkClick}
                  className={`text-2xl font-black uppercase tracking-[0.2em] py-5 border-b border-[#BBF7D0]/10 ${
                    path === '/admin' ? 'text-[#16A34A]' : 'text-[#052E16]/40'
                  }`}
                >
                  Admin Panel
                </Link>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

export default Header;

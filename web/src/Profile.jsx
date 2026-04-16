import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from './AuthContext';
import { User, Mail, Shield, Save, LogOut, ChevronRight } from 'lucide-react';

export default function Profile() {
  const { user, logout, updateProfile } = useAuth();
  const [formData, setFormData] = useState({ name: user?.name || '', email: user?.email || '' });
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setSuccess(false);

    try {
      const res = await fetch('http://127.0.0.1:5000/api/auth/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: user.id, ...formData })
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.message);
      
      updateProfile(formData);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-12 max-w-[800px] transition-colors duration-300"
    >
      <header className="flex justify-between items-center">
        <div>
           <h1 className="text-3xl font-bold tracking-tight mb-2 text-[var(--text-primary)]">Account Settings</h1>
           <p className="text-[var(--text-tertiary)] text-sm font-medium uppercase tracking-widest text-[0.7rem]">Manage your Neural Profile and Security</p>
        </div>
        <button 
          onClick={logout}
          className="flex items-center gap-2 text-red-500 font-bold text-sm hover:bg-red-500/10 px-4 py-2 rounded-xl transition-all"
        >
          <LogOut size={16}/> Sign Out
        </button>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
        {/* Profile Card */}
        <div className="md:col-span-1 space-y-6">
           <div className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-2xl p-8 text-center shadow-sm">
              <div className="w-24 h-24 bg-[#10b981] mx-auto rounded-3xl flex items-center justify-center text-white text-3xl font-black mb-6 shadow-xl shadow-green-500/20">
                {user?.name?.charAt(0) || 'U'}
              </div>
              <h3 className="font-bold text-lg text-[var(--text-primary)]">{user?.name}</h3>
              <p className="text-xs font-medium text-[var(--text-tertiary)] mt-1">{user?.email}</p>
              <div className="mt-8 pt-8 border-t border-[var(--border-color)] flex flex-col gap-3">
                 <div className="flex justify-between items-center text-[10px] font-bold text-[var(--text-tertiary)] uppercase tracking-widest">
                    <span>Member Since</span>
                    <span className="text-[var(--text-primary)]">Oct 2026</span>
                 </div>
                 <div className="flex justify-between items-center text-[10px] font-bold text-[var(--text-tertiary)] uppercase tracking-widest">
                    <span>Rank</span>
                    <span className="text-[#10b981]">Pro User</span>
                 </div>
              </div>
           </div>
        </div>

        {/* Edit Form */}
        <div className="md:col-span-2 space-y-8">
           <form onSubmit={handleSubmit} className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-[2rem] p-10 shadow-sm space-y-8">
              <div className="space-y-6">
                 <div className="space-y-2">
                    <label className="text-[0.7rem] font-bold uppercase tracking-widest text-[var(--text-tertiary)]">Public Name</label>
                    <div className="relative group">
                       <User size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500 group-focus-within:text-emerald-500 transition-colors" />
                       <input 
                         required
                         type="text"
                         className="w-full bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded-xl p-4 pl-12 text-sm font-medium text-[var(--text-primary)] focus:ring-1 focus:ring-emerald-500 outline-none transition-all"
                         value={formData.name}
                         onChange={(e) => setFormData({...formData, name: e.target.value})}
                       />
                    </div>
                 </div>

                 <div className="space-y-2">
                    <label className="text-[0.7rem] font-bold uppercase tracking-widest text-[var(--text-tertiary)]">Email Address</label>
                    <div className="relative group">
                       <Mail size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500 group-focus-within:text-emerald-500 transition-colors" />
                       <input 
                         required
                         type="email"
                         className="w-full bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded-xl p-4 pl-12 text-sm font-medium text-[var(--text-primary)] focus:ring-1 focus:ring-emerald-500 outline-none transition-all"
                         value={formData.email}
                         onChange={(e) => setFormData({...formData, email: e.target.value})}
                       />
                    </div>
                 </div>
              </div>

              <div className="pt-4 flex items-center justify-between">
                 <button 
                  type="submit" 
                  disabled={loading}
                  className="bg-[#10b981] text-white px-8 py-4 rounded-xl font-bold flex items-center gap-3 hover:bg-[#059669] disabled:opacity-50 transition-all shadow-lg shadow-green-500/20"
                 >
                   {loading ? 'Saving...' : <><Save size={18}/> Save Changes</>}
                 </button>
                 {success && <span className="text-emerald-500 text-sm font-bold">Profile updated successfully!</span>}
              </div>
           </form>

           {/* Security Toggles */}
           <div className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-[2rem] p-10 shadow-sm">
              <h4 className="font-bold mb-8 text-[var(--text-primary)]">Security & Privacy</h4>
              <div className="space-y-6">
                 {[
                   { label: 'Two-Factor Authentication', desc: 'Secure your neural connection with 2FA.', enabled: false },
                   { label: 'Cloud Persistence', desc: 'Automatically sync session history to the cloud.', enabled: true },
                 ].map((item) => (
                   <div key={item.label} className="flex items-center justify-between cursor-pointer group">
                      <div>
                         <p className="text-sm font-bold tracking-tight text-[var(--text-primary)]">{item.label}</p>
                         <p className="text-[10px] font-medium text-[var(--text-tertiary)]">{item.desc}</p>
                      </div>
                      <ChevronRight size={16} className="text-gray-400 group-hover:text-emerald-500 transition-all"/>
                   </div>
                 ))}
              </div>
           </div>
        </div>
      </div>
    </motion.div>
  );
}

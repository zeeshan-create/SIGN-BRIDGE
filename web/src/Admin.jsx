import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from './AuthContext';
import { Users, Shield, Trash2, Activity, Server, Database, Search, RefreshCcw, MoreVertical } from 'lucide-react';

export default function Admin() {
  const { user } = useAuth();
  const [users, setUsers] = useState([]);
  const [stats, setStats] = useState({ totalUsers: 0, totalSessions: 0, systemStatus: 'Checking...', bridgeConnected: false });
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');

  const token = localStorage.getItem('sb_token');

  const fetchData = async () => {
    setLoading(true);
    try {
      const headers = { 'Authorization': `Bearer ${token}` };
      const [uRes, sRes] = await Promise.all([
        fetch('http://127.0.0.1:5000/api/admin/users', { headers }),
        fetch('http://127.0.0.1:5000/api/admin/stats', { headers })
      ]);
      
      const uData = await uRes.json();
      const sData = await sRes.json();
      
      setUsers(uData);
      setStats(sData);
    } catch (err) {
      console.error('Failed to fetch admin data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const deleteUser = async (userId) => {
    if (!window.confirm('Are you sure you want to delete this user? This cannot be undone.')) return;
    
    try {
      const res = await fetch(`http://127.0.0.1:5000/api/admin/users/${userId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        setUsers(users.filter(u => u.id !== userId));
        setStats({...stats, totalUsers: stats.totalUsers - 1});
      }
    } catch (err) {
      alert('Delete failed');
    }
  };

  const filteredUsers = users.filter(u => 
    u.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
    u.email.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="p-12 pb-32 max-w-[1400px] mx-auto space-y-16"
    >
      <header className="flex justify-between items-end">
        <div>
           <h1 className="text-4xl font-bold tracking-tight mb-2">Neural Admin Dashboard</h1>
           <p className="text-gray-400 text-sm font-medium uppercase tracking-widest text-[0.7rem]">Platform Oversight & User Governance</p>
        </div>
        <button onClick={fetchData} className="flex items-center gap-2 bg-gray-50 text-gray-500 hover:text-emerald-500 font-bold text-[10px] uppercase tracking-widest px-6 py-3 rounded-xl transition-all border border-gray-100">
           <RefreshCcw size={14} className={loading ? 'animate-spin' : ''} /> Refresh Neural Core
        </button>
      </header>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
         {[
           { label: 'Platform Users', value: stats.totalUsers, icon: Users, color: 'text-emerald-500' },
           { label: 'Total Sessions', value: stats.totalSessions, icon: Activity, color: 'text-blue-500' },
           { label: 'AI Bridge', value: stats.bridgeConnected ? 'Online' : 'Offline', icon: Server, color: stats.bridgeConnected ? 'text-emerald-500' : 'text-red-500' },
           { label: 'System Health', value: stats.systemStatus, icon: Shield, color: 'text-emerald-500' },
         ].map((s) => (
           <div key={s.label} className="bg-white border border-gray-100 p-8 rounded-[2rem] shadow-sm">
              <div className="flex justify-between items-start mb-6">
                 <div className={`p-3 bg-gray-50 rounded-xl ${s.color}`}>
                    <s.icon size={20} />
                 </div>
                 <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              </div>
              <p className="text-[10px] font-bold text-gray-400 uppercase tracking-[0.2em] mb-1">{s.label}</p>
              <h4 className="text-2xl font-black">{s.value}</h4>
           </div>
         ))}
      </div>

      {/* User Management Section */}
      <div className="bg-white border border-gray-100 rounded-[2.5rem] shadow-sm overflow-hidden">
         <div className="p-8 border-b border-gray-50 flex justify-between items-center bg-gray-50/30">
            <h3 className="font-bold flex items-center gap-3"><Users size={18} className="text-emerald-500"/> User Management</h3>
            <div className="relative">
               <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-300" size={16} />
               <input 
                 type="text" 
                 placeholder="Search users..." 
                 className="bg-white border border-gray-200 rounded-xl py-2 px-4 pl-12 text-sm focus:ring-1 focus:ring-emerald-500 outline-none w-64 transition-all"
                 onChange={(e) => setSearchTerm(e.target.value)}
               />
            </div>
         </div>

         <div className="overflow-x-auto">
            <table className="w-full text-left">
               <thead>
                  <tr className="border-b border-gray-50 text-[10px] font-black text-gray-400 uppercase tracking-widest">
                     <th className="px-10 py-6">Identity</th>
                     <th className="px-10 py-6">Rank</th>
                     <th className="px-10 py-6">ID Token</th>
                     <th className="px-10 py-6 text-right">Actions</th>
                  </tr>
               </thead>
               <tbody className="divide-y divide-gray-50">
                  {loading ? (
                    <tr><td colSpan="4" className="p-20 text-center text-gray-300 font-bold uppercase tracking-widest">Accessing Neural Archive...</td></tr>
                  ) : filteredUsers.map((u) => (
                    <motion.tr 
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      key={u.id} 
                      className="group hover:bg-gray-50/50 transition-colors"
                    >
                       <td className="px-10 py-6">
                          <div className="flex items-center gap-4">
                             <div className="w-10 h-10 bg-gray-100 rounded-xl flex items-center justify-center text-gray-400 font-black text-xs group-hover:bg-[#10b981] group-hover:text-white transition-all">
                                {u.name.charAt(0)}
                             </div>
                             <div>
                                <p className="text-sm font-bold text-gray-900 leading-none mb-1">{u.name}</p>
                                <p className="text-[11px] font-medium text-gray-400">{u.email}</p>
                             </div>
                          </div>
                       </td>
                       <td className="px-10 py-6">
                          <span className={`text-[10px] font-black uppercase tracking-widest px-3 py-1 rounded-full ${u.role === 'admin' ? 'bg-emerald-50 text-emerald-600' : 'bg-gray-100 text-gray-500'}`}>
                             {u.role}
                          </span>
                       </td>
                       <td className="px-10 py-6 font-mono text-[10px] text-gray-300">#{u.id.toString().slice(-8)}</td>
                       <td className="px-10 py-6 text-right">
                          <button 
                            onClick={() => deleteUser(u.id)}
                            disabled={u.role === 'admin'}
                            className={`p-2 rounded-lg transition-all ${u.role === 'admin' ? 'text-gray-100 cursor-not-allowed' : 'text-gray-300 hover:text-red-500 hover:bg-red-50'}`}
                          >
                             <Trash2 size={16} />
                          </button>
                       </td>
                    </motion.tr>
                  ))}
               </tbody>
            </table>
         </div>
         {filteredUsers.length === 0 && !loading && (
           <div className="p-20 text-center">
              <p className="text-gray-300 font-bold uppercase tracking-widest text-sm">No neural signatures found matching your search.</p>
           </div>
         )}
      </div>

      <footer className="pt-10 border-t border-gray-50 flex justify-between items-center text-[10px] font-bold text-gray-300 uppercase tracking-widest">
         <div className="flex gap-8">
            <span className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-emerald-500" /> API Gateway Secure</span>
            <span className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-emerald-500" /> Database Localized</span>
         </div>
         <p>© 2026 SignBridge Operations</p>
      </footer>
    </motion.div>
  );
}

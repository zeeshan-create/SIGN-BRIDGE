import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Users, Search, MessageCircle, Globe, Shield, Zap, 
  UserPlus, MessageSquare, MapPin, Clock, Star, ChevronRight,
  Send, Image, Paperclip, Bell, Settings, ArrowUpRight,
  Activity, Wifi, Lock, CheckCircle, ThumbsUp, Share2
} from 'lucide-react';

const mockUsers = [
  { id: 1, name: 'Sarah Chen', role: 'Certified Interpreter', location: 'San Francisco, CA', status: 'online', avatar: 'SC', skills: 156, rating: 4.9 },
  { id: 2, name: 'Marcus Johnson', role: 'ASL Instructor', location: 'Detroit, MI', status: 'online', avatar: 'MJ', skills: 234, rating: 4.8 },
  { id: 3, name: 'Emily Rodriguez', role: 'Learning Enthusiast', location: 'Austin, TX', status: 'away', avatar: 'ER', skills: 45, rating: 4.7 },
  { id: 4, name: 'David Kim', role: 'Deaf Community Advocate', location: 'Seattle, WA', status: 'offline', avatar: 'DK', skills: 189, rating: 5.0 },
];

const mockForums = [
  { id: 1, title: 'Best resources for learning ASL fingerspelling?', replies: 42, lastActivity: '2 hours ago', category: 'Learning Tips' },
  { id: 2, title: 'Medical terminology in ASL - Share your knowledge', replies: 28, lastActivity: '5 hours ago', category: 'Specialized Vocab' },
  { id: 3, title: 'Upcoming virtual meetup: Conversational practice', replies: 67, lastActivity: '1 day ago', category: 'Events' },
  { id: 4, title: 'New to interpreting career - Need advice', replies: 35, lastActivity: '3 hours ago', category: 'Career' },
];

const mockMessages = [
  { id: 1, user: 'Sarah Chen', avatar: 'SC', message: 'Just finished the new tutorial on directional signs! Really helpful for beginners.', time: '10:32 AM', likes: 12 },
  { id: 2, user: 'Marcus Johnson', avatar: 'MJ', message: 'The neural indexing feature is amazing. It learned my signing pattern in just 2 days.', time: '11:15 AM', likes: 8 },
  { id: 3, user: 'Emily Rodriguez', avatar: 'ER', message: 'Anyone tried the new community feature? Made some great connections!', time: '12:45 PM', likes: 15 },
];

const securityChecks = [
  { status: 'complete', text: 'User authentication protocols verified' },
  { status: 'complete', text: 'Data encryption standards confirmed' },
  { status: 'complete', text: 'Privacy controls validated' },
  { status: 'complete', text: 'Network security scans passed' },
  { status: 'complete', text: 'Access control matrices verified' },
];

export default function CommunityView() {
  const [activeTab, setActiveTab] = useState('feed');
  const [searchQuery, setSearchQuery] = useState('');
  const [showChat, setShowChat] = useState(false);

  const tabs = [
    { id: 'feed', label: 'Community Feed', icon: MessageSquare },
    { id: 'members', label: 'Members', icon: Users },
    { id: 'forums', label: 'Discussion Forums', icon: MessageCircle },
    { id: 'events', label: 'Events', icon: Activity },
  ];

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="min-h-screen bg-[#F0FDF4] p-8 pt-24"
    >
      <div className="max-w-[1600px] mx-auto">
        {/* Header */}
        <div className="flex justify-between items-end mb-12">
          <div>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-[#16A34A]/10 rounded-2xl flex items-center justify-center text-[#16A34A]">
                <Users size={24} />
              </div>
              <span className="text-[10px] font-black text-[#16A34A] uppercase tracking-[0.4em]">Module Active</span>
            </div>
            <h1 className="text-4xl md:text-5xl font-black text-[#052E16] tracking-tight mb-2">SignBridge Hub</h1>
            <p className="text-[#16A34A]/60 text-sm font-medium">Connect with other interpreters and learners worldwide</p>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="relative">
              <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" />
              <input 
                type="text"
                placeholder="Search members..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 pr-6 py-4 bg-white border border-[#BBF7D0]/30 rounded-[1.5rem] text-sm font-medium text-[#052E16] placeholder:text-gray-300 focus:outline-none focus:border-[#16A34A]/50 w-64 shadow-sm"
              />
            </div>
            <motion.button 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-8 py-4 bg-[#16A34A] text-white rounded-[1.5rem] text-[10px] font-black uppercase tracking-widest hover:bg-[#15803d] transition-all shadow-lg shadow-[#16A34A]/20 flex items-center gap-3"
            >
              <UserPlus size={16} /> Invite
            </motion.button>
          </div>
        </div>

        {/* Security Status Banner */}
        <div className="bg-gradient-to-r from-[#052E16] to-[#052E16]/80 rounded-[2rem] p-6 mb-12 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 bg-emerald-500/20 rounded-xl flex items-center justify-center">
              <Shield size={20} className="text-emerald-400" />
            </div>
            <div>
              <h3 className="text-white font-black text-sm tracking-wide">Global Networking Protocols</h3>
              <p className="text-emerald-400/60 text-[10px] font-medium">Final security checks in progress</p>
            </div>
          </div>
          <div className="flex items-center gap-6">
            {securityChecks.slice(0, 3).map((check, i) => (
              <div key={i} className="flex items-center gap-2">
                <CheckCircle size={14} className="text-emerald-400" />
                <span className="text-emerald-400/60 text-[10px] font-medium">{check.text.split(' ')[0]}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
          {/* Left Column - Feed & Tabs */}
          <div className="lg:col-span-2 space-y-8">
            {/* Tab Navigation */}
            <div className="flex gap-2 bg-white p-2 rounded-[2rem] border border-[#BBF7D0]/30 shadow-sm">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex-1 flex items-center justify-center gap-2 py-4 rounded-[1.5rem] text-[10px] font-black uppercase tracking-widest transition-all ${
                    activeTab === tab.id 
                      ? 'bg-[#16A34A] text-white shadow-md' 
                      : 'text-gray-400 hover:bg-[#F0FDF4] hover:text-[#16A34A]'
                  }`}
                >
                  <tab.icon size={16} /> {tab.label}
                </button>
              ))}
            </div>

            {/* Content Area */}
            <AnimatePresence mode="wait">
              {activeTab === 'feed' && (
                <motion.div 
                  key="feed"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="space-y-6"
                >
                  {/* New Post Card */}
                  <div className="bg-white border border-[#BBF7D0]/30 rounded-[2.5rem] p-8 shadow-sm">
                    <div className="flex items-start gap-4 mb-6">
                      <div className="w-12 h-12 bg-[#052E16] rounded-2xl flex items-center justify-center text-white text-sm font-black">
                        YU
                      </div>
                      <div className="flex-1">
                        <input 
                          type="text"
                          placeholder="Share your ASL journey..."
                          className="w-full text-[#052E16] placeholder:text-gray-300 font-medium text-lg border-none focus:outline-none bg-transparent"
                        />
                      </div>
                    </div>
                    <div className="flex items-center justify-between pt-6 border-t border-[#F0FDF4]">
                      <div className="flex gap-4">
                        <button className="flex items-center gap-2 text-gray-400 hover:text-[#16A34A] transition-colors">
                          <Image size={18} /> <span className="text-xs font-medium">Photo</span>
                        </button>
                        <button className="flex items-center gap-2 text-gray-400 hover:text-[#16A34A] transition-colors">
                          <Paperclip size={18} /> <span className="text-xs font-medium">File</span>
                        </button>
                      </div>
                      <motion.button 
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="px-8 py-3 bg-[#16A34A] text-white rounded-[1.5rem] text-[10px] font-black uppercase tracking-widest flex items-center gap-2"
                      >
                        <Send size={14} /> Post
                      </motion.button>
                    </div>
                  </div>

                  {/* Feed Posts */}
                  {mockMessages.map((msg) => (
                    <motion.div 
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      key={msg.id}
                      className="bg-white border border-[#BBF7D0]/30 rounded-[2.5rem] p-8 shadow-sm"
                    >
                      <div className="flex items-start gap-4 mb-4">
                        <div className="w-12 h-12 bg-[#F0FDF4] rounded-2xl flex items-center justify-center text-[#16A34A] text-sm font-black">
                          {msg.avatar}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-1">
                            <span className="font-black text-[#052E16]">{msg.user}</span>
                            <span className="text-[10px] text-gray-400 font-medium">{msg.time}</span>
                          </div>
                          <p className="text-gray-600 font-medium leading-relaxed">{msg.message}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-6 pt-4 border-t border-[#F0FDF4]">
                        <button className="flex items-center gap-2 text-gray-400 hover:text-[#16A34A] transition-colors">
                          <ThumbsUp size={16} /> <span className="text-xs font-medium">{msg.likes}</span>
                        </button>
                        <button className="flex items-center gap-2 text-gray-400 hover:text-[#16A34A] transition-colors">
                          <MessageCircle size={16} /> <span className="text-xs font-medium">Reply</span>
                        </button>
                        <button className="flex items-center gap-2 text-gray-400 hover:text-[#16A34A] transition-colors">
                          <Share2 size={16} /> <span className="text-xs font-medium">Share</span>
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </motion.div>
              )}

              {activeTab === 'members' && (
                <motion.div 
                  key="members"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="grid grid-cols-1 md:grid-cols-2 gap-6"
                >
                  {mockUsers.map((user) => (
                    <div 
                      key={user.id}
                      className="bg-white border border-[#BBF7D0]/30 rounded-[2.5rem] p-8 shadow-sm hover:shadow-md transition-all cursor-pointer"
                    >
                      <div className="flex items-start gap-4 mb-6">
                        <div className="w-14 h-14 bg-[#F0FDF4] rounded-2xl flex items-center justify-center text-[#16A34A] text-lg font-black">
                          {user.avatar}
                        </div>
                        <div className="flex-1">
                          <h4 className="font-black text-[#052E16] text-lg mb-1">{user.name}</h4>
                          <p className="text-xs text-[#16A34A] font-medium">{user.role}</p>
                        </div>
                        <div className={`w-3 h-3 rounded-full ${user.status === 'online' ? 'bg-emerald-500' : user.status === 'away' ? 'bg-amber-400' : 'bg-gray-300'}`} />
                      </div>
                      <div className="flex items-center gap-4 mb-6 text-gray-400 text-xs font-medium">
                        <div className="flex items-center gap-1">
                          <MapPin size={14} /> {user.location}
                        </div>
                      </div>
                      <div className="flex items-center justify-between pt-6 border-t border-[#F0FDF4]">
                        <div className="flex items-center gap-4">
                          <span className="text-xs font-black text-gray-300 uppercase tracking-wider">{user.skills} Signs</span>
                          <div className="flex items-center gap-1 text-amber-500">
                            <Star size={14} fill="currentColor" />
                            <span className="text-xs font-black">{user.rating}</span>
                          </div>
                        </div>
                        <button className="text-[#16A34A] text-xs font-black uppercase tracking-widest flex items-center gap-1 hover:gap-2 transition-all">
                          Connect <ChevronRight size={14} />
                        </button>
                      </div>
                    </div>
                  ))}
                </motion.div>
              )}

              {activeTab === 'forums' && (
                <motion.div 
                  key="forums"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="space-y-4"
                >
                  {mockForums.map((forum) => (
                    <div 
                      key={forum.id}
                      className="bg-white border border-[#BBF7D0]/30 rounded-[2rem] p-6 shadow-sm hover:shadow-md transition-all cursor-pointer"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <div className="w-10 h-10 bg-[#F0FDF4] rounded-xl flex items-center justify-center text-[#16A34A]">
                            <MessageCircle size={18} />
                          </div>
                          <div>
                            <h4 className="font-black text-[#052E16] mb-1">{forum.title}</h4>
                            <div className="flex items-center gap-4 text-xs text-gray-400 font-medium">
                              <span className="px-2 py-1 bg-[#F0FDF4] rounded-full text-[#16A34A]">{forum.category}</span>
                              <span>{forum.replies} replies</span>
                              <span>{forum.lastActivity}</span>
                            </div>
                          </div>
                        </div>
                        <ChevronRight size={18} className="text-gray-300" />
                      </div>
                    </div>
                  ))}
                </motion.div>
              )}

              {activeTab === 'events' && (
                <motion.div 
                  key="events"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-center py-20"
                >
                  <div className="w-20 h-20 bg-[#F0FDF4] rounded-[2rem] flex items-center justify-center mx-auto mb-6 text-[#16A34A]">
                    <Activity size={40} />
                  </div>
                  <h3 className="text-2xl font-black text-[#052E16] mb-2">Upcoming Events</h3>
                  <p className="text-gray-400 font-medium">Check back soon for virtual meetups and workshops</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Right Column - Stats & Activity */}
          <div className="space-y-8">
            {/* Network Status */}
            <div className="bg-[#052E16] border border-white/5 rounded-[2.5rem] p-8">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-white font-black text-sm uppercase tracking-wider">Network Status</h3>
                <div className="flex items-center gap-2">
                  <Wifi size={14} className="text-emerald-400 animate-pulse" />
                  <span className="text-emerald-400 text-[10px] font-black uppercase tracking-widest">Connected</span>
                </div>
              </div>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400 text-xs font-medium">Active Members</span>
                  <span className="text-white font-black">1,247</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400 text-xs font-medium">Online Now</span>
                  <span className="text-emerald-400 font-black">89</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400 text-xs font-medium">Discussions</span>
                  <span className="text-white font-black">342</span>
                </div>
              </div>
            </div>

            {/* Recent Activity */}
            <div className="bg-white border border-[#BBF7D0]/30 rounded-[2.5rem] p-8">
              <h3 className="font-black text-[#052E16] text-sm uppercase tracking-wider mb-6">Recent Activity</h3>
              <div className="space-y-4">
                {[1,2,3,4].map((i) => (
                  <div key={i} className="flex items-center gap-4 p-3 rounded-xl hover:bg-[#F0FDF4] transition-colors cursor-pointer">
                    <div className="w-8 h-8 bg-[#F0FDF4] rounded-lg flex items-center justify-center text-[#16A34A]">
                      <Activity size={14} />
                    </div>
                    <div className="flex-1">
                      <p className="text-xs font-medium text-[#052E16]">New member joined the community</p>
                      <span className="text-[10px] text-gray-400">{i * 5} minutes ago</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-gradient-to-br from-[#16A34A]/10 to-[#16A34A]/5 border border-[#BBF7D0]/30 rounded-[2.5rem] p-8">
              <h3 className="font-black text-[#052E16] text-sm uppercase tracking-wider mb-6">Quick Actions</h3>
              <div className="space-y-3">
                <button className="w-full py-4 bg-white border border-[#BBF7D0]/30 rounded-2xl text-[#052E16] text-xs font-black uppercase tracking-widest hover:bg-[#16A34A] hover:text-white transition-all flex items-center justify-between px-6">
                  <span>Find a Study Partner</span>
                  <ArrowUpRight size={14} />
                </button>
                <button className="w-full py-4 bg-white border border-[#BBF7D0]/30 rounded-2xl text-[#052E16] text-xs font-black uppercase tracking-widest hover:bg-[#16A34A] hover:text-white transition-all flex items-center justify-between px-6">
                  <span>Join a Discussion</span>
                  <ArrowUpRight size={14} />
                </button>
                <button className="w-full py-4 bg-white border border-[#BBF7D0]/30 rounded-2xl text-[#052E16] text-xs font-black uppercase tracking-widest hover:bg-[#16A34A] hover:text-white transition-all flex items-center justify-between px-6">
                  <span>Share Resources</span>
                  <ArrowUpRight size={14} />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
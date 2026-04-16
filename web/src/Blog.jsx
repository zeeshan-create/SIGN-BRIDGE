import React from 'react';
import { motion } from 'framer-motion';
import { BookOpen, Calendar, User, ArrowRight } from 'lucide-react';

function Blog() {
  const posts = [
    {
      title: "The Evolution of ASL Recognition Technology",
      excerpt: "From early computer vision experiments to modern neural networks, explore how ASL detection has advanced over the decades.",
      date: "2024-04-15",
      author: "Dr. Sarah Chen",
      readTime: "5 min read"
    },
    {
      title: "Improving Accuracy: Challenges in Real-World ASL Detection",
      excerpt: "Lighting conditions, hand orientations, and cultural variations present unique challenges for accurate ASL interpretation.",
      date: "2024-04-10",
      author: "Mike Johnson",
      readTime: "7 min read"
    },
    {
      title: "Bridging the Communication Gap: ASL in Education",
      excerpt: "How AI-powered ASL tools are revolutionizing accessibility in classrooms and online learning environments.",
      date: "2024-04-05",
      author: "Emma Rodriguez",
      readTime: "6 min read"
    }
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.15, ease: [0.22, 1, 0.36, 1] }}
      className="min-h-screen bg-[#f8fafc] font-inter pb-32"
    >
      {/* Hero Section */}
      <section className="bg-white border-b border-emerald-50 pt-32 pb-24 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-gradient-to-bl from-emerald-100/30 to-transparent rounded-full blur-[100px] pointer-events-none" />
        <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-gradient-to-tr from-emerald-50/50 to-transparent rounded-full blur-[100px] pointer-events-none" />

        <div className="max-w-[1200px] mx-auto px-6 text-center relative z-10">
          <motion.div initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-50 text-emerald-700 rounded-full text-[10px] font-black uppercase tracking-widest mb-8 border border-emerald-100">
            <BookOpen size={14} /> Knowledge Hub
          </motion.div>
          <motion.h1 initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.02 }} className="text-6xl md:text-7xl font-black tracking-tight text-[#052E16] mb-8 leading-[1.1]">
            Insights & <br /> <span className="text-[#10b981]">Innovations</span>
          </motion.h1>
          <p className="text-xl text-gray-500 font-medium leading-relaxed max-w-2xl mx-auto">
            Deep dives into ASL technology, accessibility advancements, and the future of sign language communication.
          </p>
        </div>
      </section>

      {/* Blog Posts */}
      <section className="max-w-[1000px] mx-auto px-6 py-24">
        <div className="space-y-12">
          {posts.map((post, index) => (
            <motion.article
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="bg-white/60 backdrop-blur-md rounded-[2.5rem] border border-gray-100 shadow-sm hover:shadow-lg transition-all duration-300 p-8 md:p-12 group cursor-pointer"
            >
              <div className="flex items-start justify-between mb-6">
                <div className="flex items-center gap-4 text-sm text-gray-400 font-medium">
                  <div className="flex items-center gap-2">
                    <Calendar size={16} />
                    <span>{new Date(post.date).toLocaleDateString()}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <User size={16} />
                    <span>{post.author}</span>
                  </div>
                </div>
                <span className="text-xs font-black uppercase tracking-widest text-emerald-500 bg-emerald-50 px-3 py-1 rounded-full border border-emerald-100">
                  {post.readTime}
                </span>
              </div>

              <h2 className="text-2xl md:text-3xl font-black text-[#052E16] mb-4 group-hover:text-[#10b981] transition-colors">
                {post.title}
              </h2>

              <p className="text-gray-500 font-medium leading-relaxed mb-6">
                {post.excerpt}
              </p>

              <div className="flex items-center gap-2 text-emerald-600 font-bold uppercase tracking-widest text-sm group-hover:gap-4 transition-all">
                <span>Read Article</span>
                <ArrowRight size={16} />
              </div>
            </motion.article>
          ))}
        </div>
      </section>

      {/* Newsletter Signup */}
      <section className="max-w-[1200px] mx-auto px-6 pb-24">
        <div className="bg-[#052E16] rounded-[4.5rem] p-16 md:p-24 text-center relative overflow-hidden shadow-2xl shadow-[#052E16]/20">
          <div className="absolute top-0 right-0 w-[400px] h-[400px] bg-emerald-500 rounded-full blur-[120px] opacity-20 pointer-events-none" />
          <h3 className="text-4xl md:text-5xl font-black text-white mb-8 relative z-10 tracking-tight">
            Stay Updated
          </h3>
          <p className="text-emerald-100/60 text-xl font-medium mb-16 max-w-xl mx-auto relative z-10 leading-relaxed">
            Get the latest insights on ASL technology and accessibility innovations delivered to your inbox.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 max-w-md mx-auto relative z-10">
            <input
              type="email"
              placeholder="Enter your email"
              className="flex-1 px-6 py-4 rounded-2xl border border-emerald-100/20 bg-white/10 text-white placeholder-emerald-100/50 focus:outline-none focus:border-emerald-300 transition-colors"
            />
            <button className="px-8 py-4 bg-[#10b981] text-[#052E16] rounded-2xl font-black uppercase tracking-widest hover:bg-white transition-colors">
              Subscribe
            </button>
          </div>
        </div>
      </section>
    </motion.div>
  );
}

export default Blog;
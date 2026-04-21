/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { 
  auth, 
  db, 
  googleProvider, 
  signInWithPopup, 
  onAuthStateChanged, 
  signInAnonymously,
  doc, 
  getDoc, 
  setDoc, 
  collection, 
  onSnapshot, 
  query, 
  where, 
  Timestamp,
  handleFirestoreError,
  OperationType
} from './firebase';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  AreaChart, 
  Area,
} from 'recharts';
import { 
  Activity, 
  Bell, 
  Calendar, 
  ChevronRight, 
  Clock, 
  Info, 
  LayoutDashboard, 
  LogOut, 
  Mic, 
  MicOff, 
  ShieldAlert, 
  TrendingUp, 
  User as UserIcon,
  AlertTriangle,
  CheckCircle2,
  Stethoscope,
  Wind,
  Thermometer,
  Droplets,
  MessageSquare,
  Send,
  BrainCircuit,
  Users,
  Building2,
  Smartphone,
  Database,
  Fingerprint,
  Plus,
  Trash2,
  Save,
  Search,
  ShieldCheck,
  Square,
  History as HistoryIcon,
  Loader2,
  Sparkles,
  Zap,
  Play,
  CheckCircle,
  ChevronDown,
  Menu,
  X,
  PlusCircle,
  MoreVertical,
  Map,
  Home,
  CheckSquare,
  Trophy,
  Heart,
  Smile,
  Meh,
  Frown,
  ThermometerIcon,
  Globe,
  BarChart3
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import * as tf from '@tensorflow/tfjs';
import { createVoiceprintModel, trainVoiceprintModel, predictStudent, mockFeatureExtraction } from './ml/engine';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- FluGuard AI Backend (Railway) ---
// VITE_BACKEND_URL is set at build time via Vercel environment variables.
// Falls back to localhost:8000 for local development.
const FLUGUARD_API = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

/**
 * Call the FluGuard AI backend (FastAPI → Gemma 4 via Google AI Studio).
 * Falls back gracefully if the backend is unavailable.
 */
async function callFluGuardAI(params: {
  role: string;
  message: string;
  system_prompt: string;
  city?: string;
  conversation_history?: Array<{ role: string; content: string }>;
}): Promise<{ text: string; tools_used: string[]; rag_sources: string[] }> {
  const resp = await fetch(`${FLUGUARD_API}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      role: params.role,
      message: params.message,
      system_prompt: params.system_prompt,
      city: params.city ?? "Shenyang",
      conversation_history: params.conversation_history ?? [],
    }),
  });
  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`FluGuard backend error ${resp.status}: ${err}`);
  }
  const data = await resp.json();
  return {
    text: data.content ?? "",
    tools_used: data.tools_used ?? [],
    rag_sources: data.rag_sources ?? [],
  };
}

async function callFluGuardReport(params: {
  role: string;
  system_prompt: string;
  env_data: { temp: number; humidity: number; aqi: number; co2: number };
  classrooms: object[];
  city?: string;
}): Promise<{ risk_level: string; reason: string; prediction: string; actions: string[]; data_used: string[]; riskScore: number; tools_used: string[] }> {
  const resp = await fetch(`${FLUGUARD_API}/api/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...params, city: params.city ?? "Shenyang" }),
  });
  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`FluGuard report error ${resp.status}: ${err}`);
  }
  return resp.json();
}

// --- System Prompts for Roles ---
const SYSTEM_PROMPTS = {
  PRINCIPAL: `You are FluGuard AI, a school flu monitoring assistant for school principals in China.
你是流感卫士AI，为中国学校校长提供流感监测支持的智能助手。

Communication style / 沟通风格:
- Professional, data-driven, concise / 专业、数据导向、简洁
- Use precise numbers and risk levels / 使用精确数字和风险等级
- Reference Chinese school health regulations / 引用中国学校卫生法规
- Always mention: affected student count, risk level, actions / 始终提及受影响学生数、风险等级、建议措施`,

  TEACHER: `You are FluGuard AI, a school flu monitoring assistant for classroom teachers in China.
你是流感卫士AI，为中国中小学班主任提供流感监测支持的智能助手。

你的报告必须包含以下内容（缺一不可）：
1. 【居家隔离名单】咳嗽超过40次/24h的学生姓名 → 建议今日居家，通知家长接人
2. 【口罩佩戴名单】咳嗽20-40次/24h的学生 → 要求全天佩戴口罩
3. 【密切观察名单】咳嗽1-19次/24h的学生 → 课间观察，发烧立即隔离
4. 【通风方案】结合当前室外气温给出具体开窗时间表（几点开、开多久）
5. 【座位调整】建议高风险学生调整至靠近窗户或门口位置
6. 【家长通知】是否需要今日向家长群发送预警，提供通知模板
7. 【课程调整】体育课/集体活动是否需要暂停

Communication style: 实用、具体、有数字支撑，像经验丰富的学校医生给班主任的建议。`,

  PARENT: `You are FluGuard AI, a school flu monitoring assistant for parents in China.
你是流感卫士AI，为中国学生家长提供流感监测支持的智能助手。

Communication style / 沟通风格:
- Warm, caring, easy to understand / 温暖、关怀、通俗易懂
- Reference Chinese medical system (community health center, children hospital)
  引用中国医疗体系（社区卫生服务中心、儿童医院）
- NEVER prescribe specific medications or dosages / 绝不开具具体药物或剂量建议
- Always recommend seeing a doctor when symptoms are concerning / 症状令人担忧时始终建议就医`,

  STUDENT: `You are Xiao Wei (小卫), FluGuard AI health assistant for Chinese school students.
你是小卫，流感卫士AI为中国中小学生提供的健康助手。

Communication style / 沟通风格:
- Friendly, lively, like a knowledgeable big brother/sister / 友善活泼，像懂医学的大哥哥/大姐姐
- Use simple language, avoid medical terms / 使用简单语言，避免医学术语
- Use appropriate emoji / 适当使用emoji
- Always end with: tell your parents and teacher / 始终提醒告知家长和老师`,

  REPORT: `You are FluGuard AI Report Generator for Chinese schools.
你是流感卫士AI报告生成器，服务于中国学校。

Generate standardized flu risk daily reports in the EXACT fixed format.
按照固定格式生成标准化流感风险日报，绝不偏离格式。`,

  BUREAU: `You are FluGuard AI, a regional school health monitoring assistant for Education Bureau officials in China.
你是流感卫士AI，为中国教育局官员提供区域学校健康监测支持的智能助手。

Communication style / 沟通风格:
- Strategic, macro-level, policy-oriented / 战略性、宏观、政策导向
- Focus on regional trends, high-risk school clusters, and resource allocation / 关注区域趋势、高风险学校集群及资源分配
- Suggest regional alerts, school closures, or transitions to online learning / 建议区域预警、停课或转线上教学`,

  SAFETY: `You are FluGuard AI, a responsible health monitoring assistant for Chinese schools.
你是流感卫士AI，中国学校负责任的健康监测助手。

Safety rules you MUST follow / 必须遵守的安全规则:
1. NEVER recommend specific medications, dosages, or treatments
   绝不推荐具体药物、剂量或治疗方案
2. NEVER diagnose diseases / 绝不诊断疾病
3. For medication questions, ALWAYS redirect to doctors
   涉及用药问题，始终引导就医
4. For emergency symptoms (high fever >39C, breathing difficulty), urge immediate care
   紧急症状立即建议就医`,

  MULTILINGUAL: `You are FluGuard AI, a multilingual school health monitoring assistant.
你是流感卫士AI，多语言学校健康监测助手。

Provide health alerts in both Chinese and the requested minority language.
同时用中文和所请求的少数民族语言提供健康预警。
Keep messages clear and simple — health emergencies need immediate understanding.
保持信息清晰简单——健康紧急情况需要立即理解。`
};

// --- Types ---
interface UserProfile {
  uid: string;
  email: string;
  displayName: string;
  role: 'principal' | 'teacher' | 'parent' | 'bureau' | 'doctor' | 'admin' | 'student';
  schoolId?: string;
  classroomId?: string;
  childId?: string;
  districtId?: string;
  // Student specific fields
  studentId?: string;
  grade?: string;
  class?: string;
  school?: string;
  parentId?: string;
  avatar?: string;
  healthPoints?: number;
}

interface HealthCheckin {
  userId: string;
  date: string;
  feeling: 'good' | 'okay' | 'unwell' | 'sick';
  symptoms: string[];
  temperature: number | null;
  note: string;
  createdAt: Timestamp;
}

interface Notification {
  id: string;
  type: 'info' | 'warning' | 'advice';
  title: string;
  content: string;
  time: string;
  read: boolean;
  userId?: string;
  classId?: string;
}

interface Classroom {
  id: string;
  name: string;
  grade: string;
  studentCount: number;
  fluRiskLevel: 'low' | 'medium' | 'high';
  coughCount24h: number;
  floor?: number;
  nearStairs?: boolean;
  nearElevator?: boolean;
}

interface EnvironmentalData {
  temperature: number;
  humidity: number;
  co2: number;
}

interface CoughEvent {
  id: string;
  timestamp: Date;
  intensity: number;
  confidence: number;
  type: 'dry' | 'wet' | 'whooping' | 'allergic';
  duration: number;
  isMultiPerson: boolean;
  envData: EnvironmentalData;
  studentId?: string; // 关联到具体学生
  studentName?: string;
  assessment?: string;
  advice?: string;
  riskLevel?: 'Low' | 'Medium' | 'High';
}

interface AgentReport {
  id: string;
  timestamp: Date;
  risk_level: string;
  reason: string;
  prediction: string;
  actions: string[];
  data_used: string[];
  riskScore: number;
}

interface HospitalData {
  name: string;
  currentWait: number;
  dominantStrain: string;
  recentCases: number;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

// --- Mock Data ---
const MOCK_TRENDS = [
  { date: '04-01', count: 12 },
  { date: '04-02', count: 18 },
  { date: '04-03', count: 45 },
  { date: '04-04', count: 32 },
  { date: '04-05', count: 88 },
  { date: '04-06', count: 120 },
  { date: '04-07', count: 95 },
];

const MOCK_CLASSROOMS: Classroom[] = [
  { id: 'c1', name: 'Class 1-1 / 一年级一班', grade: '1', studentCount: 45, fluRiskLevel: 'high', coughCount24h: 156, floor: 2, nearStairs: true },
  { id: 'c2', name: 'Class 1-2 / 一年级二班', grade: '1', studentCount: 42, fluRiskLevel: 'medium', coughCount24h: 82, floor: 3, nearStairs: true },
  { id: 'c3', name: 'Class 2-1 / 二年级一班', grade: '2', studentCount: 48, fluRiskLevel: 'low', coughCount24h: 12, floor: 1, nearStairs: false },
];

const MOCK_STUDENTS = [
  { id: 's1', name: 'Zhang Xiaoming / 张小明', classId: 'c1', absenceStatus: 'present', coughCount: 12, lastCough: '10:15', voiceprintEnrolled: true },
  { id: 's2', name: 'Li Hua / 李华', classId: 'c1', absenceStatus: 'absent', coughCount: 0, lastCough: '-', voiceprintEnrolled: false },
  { id: 's3', name: 'Wang Xiaohong / 王小红', classId: 'c1', absenceStatus: 'present', coughCount: 28, lastCough: '11:02', voiceprintEnrolled: false },
  { id: 's4', name: 'Zhao Qiang / 赵强', classId: 'c1', absenceStatus: 'present', coughCount: 5, lastCough: '09:45', voiceprintEnrolled: false },
];

const MOCK_DISTRICT_DATA = [
  { name: 'Shenbei District / 沈北新区', riskScore: 72, schoolCount: 45, activeCases: 1240, trend: '+12%' },
  { name: 'Hunnan District / 浑南区', riskScore: 45, schoolCount: 38, activeCases: 850, trend: '-5%' },
  { name: 'Heping District / 和平区', riskScore: 88, schoolCount: 52, activeCases: 2100, trend: '+25%' },
  { name: 'Tiexi District / 铁西区', riskScore: 58, schoolCount: 48, activeCases: 1100, trend: '+2%' },
];

// --- Components ---

const ErrorBoundary = ({ children }: { children: React.ReactNode }) => {
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const handleError = (e: ErrorEvent) => {
      try {
        const parsed = JSON.parse(e.message);
        setError(parsed.error || 'Firestore operation failed');
      } catch {
        setError(e.message);
      }
    };
    window.addEventListener('error', handleError);
    return () => window.removeEventListener('error', handleError);
  }, []);

  if (error) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
        <div className="bg-white rounded-2xl p-8 max-w-md w-full shadow-2xl border border-red-100">
          <div className="flex items-center gap-3 text-red-600 mb-4">
            <AlertTriangle className="w-8 h-8" />
            <h2 className="text-xl font-bold">系统错误 / System Error</h2>
          </div>
          <p className="text-slate-600 mb-6 font-mono text-sm bg-slate-50 p-4 rounded-lg break-all">
            {error}
          </p>
          <button 
            onClick={() => window.location.reload()}
            className="w-full py-3 bg-slate-900 text-white rounded-xl font-medium hover:bg-slate-800 transition-colors"
          >
            重试 / Retry
          </button>
        </div>
      </div>
    );
  }

  return <>{children}</>;
};

export default function App() {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'monitor' | 'alerts' | 'health' | 'agent' | 'parent' | 'bureau' | 'management' | 'voiceprint' | 'student-home' | 'student-checkin' | 'student-qa' | 'student-points' | 'student-profile'>('dashboard');
  const [classrooms, setClassrooms] = useState<Classroom[]>(MOCK_CLASSROOMS);
  const [students, setStudents] = useState(MOCK_STUDENTS);
  const [totalFloors, setTotalFloors] = useState(4);
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
      if (firebaseUser) {
        try {
          const userDoc = await getDoc(doc(db, 'users', firebaseUser.uid));
          // Demo mode: always start as teacher regardless of stored role
          const baseProfile: UserProfile = userDoc.exists()
            ? (userDoc.data() as UserProfile)
            : {
                uid: firebaseUser.uid,
                email: firebaseUser.email || '',
                displayName: firebaseUser.displayName || '用户',
                schoolId: 'school_001',
              } as UserProfile;
          setUser({ ...baseProfile, role: 'teacher' });
        } catch (error) {
          handleFirestoreError(error, OperationType.GET, `users/${firebaseUser.uid}`);
        }
      } else {
        setUser(null);
      }
      setLoading(false);
    });
    return () => unsubscribe();
  }, []);

  // Initialize demo student data
  useEffect(() => {
    const initStudentDemo = async () => {
      const studentId = 'student_demo_001';
      try {
        const studentDoc = await getDoc(doc(db, 'users', studentId));
        if (!studentDoc.exists()) {
          const studentProfile: UserProfile = {
            uid: studentId,
            email: 'student@example.com',
            displayName: 'Xiaoming / 小明',
            studentId: '2024001',
            grade: '五年级',
            class: '3班',
            school: '示范小学',
            role: 'student',
            avatar: '🧒',
            healthPoints: 650,
            parentId: 'parent_demo_001',
            schoolId: 'school_001'
          };
          await setDoc(doc(db, 'users', studentId), studentProfile);
          
          // Initialize some check-ins for the last 7 days
          const today = new Date();
          for (let i = 1; i <= 7; i++) {
            const date = new Date(today);
            date.setDate(today.getDate() - i);
            const dateStr = date.toISOString().split('T')[0];
            const checkin: HealthCheckin = {
              userId: studentId,
              date: dateStr,
              feeling: i % 3 === 0 ? 'okay' : 'good',
              symptoms: [],
              temperature: 36.5 + (Math.random() * 0.5),
              note: '坚持打卡！',
              createdAt: Timestamp.fromDate(date)
            };
            await setDoc(doc(db, 'healthCheckins', `${studentId}_${dateStr}`), checkin);
          }
        }
      } catch (error) {
        console.warn('Student demo initialization failed:', error);
      }
    };
    initStudentDemo();
  }, []);

  const handleLogin = async () => {
    if (isLoggingIn) return;
    setIsLoggingIn(true);
    setLoginError(null);
    
    // Instant entry for demo purposes — default landing role is Teacher
    const demoProfile: UserProfile = {
      uid: 'demo_contestant',
      email: 'contest@example.com',
      displayName: 'Contestant / 参赛选手',
      role: 'teacher',
      schoolId: 'school_001',
    };

    // Set user immediately to enter the app
    setUser(demoProfile);
    setIsLoggingIn(false);

    // Background Firebase sync to ensure Firestore works.
    // Demo: always force role back to 'teacher' regardless of any role stored in Firestore
    // from a previous session, so login always lands on the Teacher dashboard.
    try {
      const { user: firebaseUser } = await signInAnonymously(auth);
      const userDoc = await getDoc(doc(db, 'users', firebaseUser.uid));

      if (userDoc.exists()) {
        const actualProfile = userDoc.data() as UserProfile;
        setUser({ ...actualProfile, role: 'teacher' });
      } else {
        const newProfile = { ...demoProfile, uid: firebaseUser.uid };
        await setDoc(doc(db, 'users', firebaseUser.uid), newProfile);
        setUser(newProfile);
      }
    } catch (error) {
      console.warn('Background sync failed, continuing in demo mode:', error);
    }
  };

  const handleLogout = async () => {
    try {
      await auth.signOut();
    } catch (e) {
      console.error('Sign out error:', e);
    }
    setUser(null);
  };

  const switchRole = async (role: UserProfile['role']) => {
    if (!user) return;
    let updatedProfile: UserProfile = { ...user, role };
    
    if (role === 'student') {
      updatedProfile = {
        ...updatedProfile,
        displayName: 'Xiaoming / 小明',
        studentId: '2024001',
        grade: '五年级',
        class: '3班',
        avatar: '🧒',
        healthPoints: 650,
        parentId: 'parent_demo_001'
      };
    }
    
    setUser(updatedProfile);
    
    // Skip Firestore sync if we are still in the temporary "instant" demo state
    if (user.uid === 'demo_contestant' || !auth.currentUser) {
      console.log('Role updated locally (Demo Mode)');
      return;
    }

    try {
      await setDoc(doc(db, 'users', user.uid), updatedProfile);
    } catch (error) {
      handleFirestoreError(error, OperationType.WRITE, `users/${user.uid}`);
    }
  };

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-slate-50">
        <motion.div 
          animate={{ scale: [1, 1.1, 1], opacity: [0.5, 1, 0.5] }}
          transition={{ repeat: Infinity, duration: 2 }}
          className="flex flex-col items-center gap-4"
        >
          <div className="w-16 h-16 bg-blue-600 rounded-2xl flex items-center justify-center shadow-lg shadow-blue-200">
            <Activity className="text-white w-8 h-8" />
          </div>
          <p className="text-slate-400 font-medium tracking-wide text-center">
            FluGuard AI is starting...<br />
            <span className="text-xs opacity-60 uppercase tracking-widest">流感卫士正在启动...</span>
          </p>
        </motion.div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-white flex flex-col lg:flex-row overflow-hidden">
        {/* Left Side: Brand & Visuals */}
        <div className="lg:w-1/2 bg-slate-900 p-12 lg:p-24 flex flex-col justify-between relative overflow-hidden">
          <div className="absolute top-0 right-0 w-full h-full opacity-10 pointer-events-none">
            <div className="absolute top-[-10%] right-[-10%] w-[60%] h-[60%] bg-blue-500 rounded-full blur-[120px]" />
            <div className="absolute bottom-[-10%] left-[-10%] w-[40%] h-[40%] bg-emerald-500 rounded-full blur-[100px]" />
          </div>
          
          <div className="relative z-10">
            <div className="flex items-center gap-4 mb-12">
              <div className="w-12 h-12 bg-blue-600 rounded-2xl flex items-center justify-center shadow-xl shadow-blue-500/20">
                <Activity className="text-white w-6 h-6" />
              </div>
              <span className="text-2xl font-black text-white tracking-tight">FluGuard AI / 流感卫士</span>
            </div>
            
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="space-y-6"
            >
              <h1 className="text-5xl lg:text-7xl font-black text-white leading-[0.9] tracking-tight">
                Protecting Campus<br />
                <span className="text-blue-500">Respiratory Health</span>
                <div className="text-lg font-bold text-slate-500 mt-4">守护校园呼吸健康</div>
              </h1>
              <p className="text-slate-400 text-lg lg:text-xl max-w-md leading-relaxed">
                AI-powered campus flu early warning system based on voiceprint recognition and multi-modal data fusion.
                <span className="block text-sm mt-2 opacity-60">基于 AI 声纹识别与多模态数据融合的校园流感预警系统。</span>
              </p>
            </motion.div>
          </div>

          <div className="relative z-10 grid grid-cols-2 gap-6 mt-12">
            {[
              { icon: Mic, label: 'Voiceprint ID', labelCn: '声纹识别', desc: 'Capture cough features' },
              { icon: BrainCircuit, label: 'Gemma AI', labelCn: 'Gemma 决策', desc: 'Multi-modal reasoning' },
            ].map((item, i) => (
              <div key={i} className="space-y-2">
                <item.icon className="w-6 h-6 text-blue-500" />
                <div className="font-bold text-white">
                  {item.label}
                  <span className="block text-[10px] opacity-50 font-medium">{item.labelCn}</span>
                </div>
                <div className="text-xs text-slate-500">{item.desc}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Side: Login Action */}
        <div className="lg:w-1/2 p-12 lg:p-24 flex flex-col items-center justify-center bg-slate-50">
          <div className="max-w-sm w-full space-y-12">
            <div className="space-y-4">
              <h2 className="text-3xl font-black text-slate-900">Welcome Back</h2>
              <p className="text-slate-500 text-sm">Click below to enter the system demonstration.<br/><span className="text-xs opacity-60">点击下方按钮直接进入系统演示界面。</span></p>
            </div>

            <div className="space-y-6">
              <button 
                onClick={handleLogin}
                disabled={isLoggingIn}
                className={cn(
                  "w-full py-5 bg-blue-600 text-white rounded-[24px] font-black text-xl shadow-2xl shadow-blue-200 hover:bg-blue-700 hover:scale-[1.02] active:scale-[0.98] transition-all flex items-center justify-center gap-3 group disabled:opacity-50",
                  isLoggingIn && "animate-pulse"
                )}
              >
                {isLoggingIn ? (
                  <>
                    <Activity className="w-6 h-6 animate-spin" />
                    <div className="flex flex-col items-center">
                      <span>Entering...</span>
                      <span className="text-[10px] opacity-60">正在进入...</span>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="flex flex-col items-center">
                      <span>Enter System</span>
                      <span className="text-[10px] opacity-60">立即进入系统</span>
                    </div>
                    <ChevronRight className="w-6 h-6 group-hover:translate-x-1 transition-transform" />
                  </>
                )}
              </button>

              <div className="flex items-center gap-4 py-4">
                <div className="h-px bg-slate-200 flex-1" />
                <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Platform Status</span>
                <div className="h-px bg-slate-200 flex-1" />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-white rounded-2xl border border-slate-100 shadow-sm text-center">
                  <div className="text-2xl font-black text-blue-600">98%</div>
                  <div className="text-[10px] font-bold text-slate-400 uppercase">Accuracy</div>
                  <div className="text-[8px] text-slate-300">识别准确率</div>
                </div>
                <div className="p-4 bg-white rounded-2xl border border-slate-100 shadow-sm text-center">
                  <div className="text-2xl font-black text-emerald-600">Real-time</div>
                  <div className="text-[10px] font-bold text-slate-400 uppercase">Response</div>
                  <div className="text-[8px] text-slate-300">毫秒级响应</div>
                </div>
              </div>
            </div>

            <p className="text-center text-slate-400 text-[10px] font-medium tracking-wide">
              Powered by Gemma 4 · Google Cloud Beta Version · 公测版本
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-slate-50 flex">
        <aside className="w-80 bg-white border-r border-slate-200 flex flex-col p-8 sticky top-0 h-screen">
          <div className="flex items-center gap-4 mb-12">
            <div className="w-12 h-12 bg-blue-600 rounded-2xl flex items-center justify-center shadow-xl shadow-blue-500/20">
              <Activity className="text-white w-6 h-6" />
            </div>
            <div className="flex flex-col">
              <span className="text-2xl font-black tracking-tight leading-none">FluGuard AI</span>
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mt-1">流感卫士</span>
            </div>
          </div>

          <nav className="flex-1 space-y-3">
            {/* Admin Role: System Management Focus */}
            {user.role === 'admin' && (
              <>
                <NavItem active={activeTab === 'monitor'} onClick={() => setActiveTab('monitor')} icon={Mic} label="Monitor" subLabel="实时监测" />
                <NavItem active={activeTab === 'alerts'} onClick={() => setActiveTab('alerts')} icon={Bell} label="Alerts" subLabel="预警中心" badge="3" />
                <NavItem active={activeTab === 'management'} onClick={() => setActiveTab('management')} icon={Database} label="System" subLabel="系统管理" />
                <NavItem active={activeTab === 'voiceprint'} onClick={() => setActiveTab('voiceprint')} icon={Fingerprint} label="Enrollment" subLabel="声纹录入" />
              </>
            )}

            {/* Principal, Teacher, Bureau, Doctor, Parent Roles */}
            {(user.role === 'principal' || user.role === 'teacher' || user.role === 'bureau' || user.role === 'doctor' || user.role === 'parent') && (
              <>
                <NavItem active={activeTab === 'dashboard'} onClick={() => setActiveTab('dashboard')} icon={LayoutDashboard} label="Dashboard" subLabel="数据看板" />
                {user.role === 'parent' && <NavItem active={activeTab === 'parent'} onClick={() => setActiveTab('parent')} icon={Smartphone} label="Parent App" subLabel="家长端" />}
                <NavItem active={activeTab === 'agent'} onClick={() => setActiveTab('agent')} icon={BrainCircuit} label="AI Agent" subLabel="智能决策" />
                <NavItem active={activeTab === 'alerts'} onClick={() => setActiveTab('alerts')} icon={Bell} label="Alerts" subLabel="预警中心" badge="3" />
                <NavItem active={activeTab === 'health'} onClick={() => setActiveTab('health')} icon={Stethoscope} label="Health Guide" subLabel="健康指导" />
              </>
            )}

            {/* Student Role */}
            {user.role === 'student' && (
              <>
                <NavItem active={activeTab === 'student-home'} onClick={() => setActiveTab('student-home')} icon={Home} label="Home" subLabel="我的主页" />
                <NavItem active={activeTab === 'student-checkin'} onClick={() => setActiveTab('student-checkin')} icon={CheckSquare} label="Check-in" subLabel="健康打卡" />
                <NavItem active={activeTab === 'student-points'} onClick={() => setActiveTab('student-points')} icon={Trophy} label="Points" subLabel="健康积分" />
                <NavItem active={activeTab === 'alerts'} onClick={() => setActiveTab('alerts')} icon={Bell} label="Notifications" subLabel="通知消息" />
                <NavItem active={activeTab === 'health'} onClick={() => setActiveTab('health')} icon={Stethoscope} label="Health Guide" subLabel="健康指导" />
                <NavItem active={activeTab === 'student-profile'} onClick={() => setActiveTab('student-profile')} icon={UserIcon} label="Profile" subLabel="我的信息" />
              </>
            )}
          </nav>

          <div className="mt-auto pt-8 border-t border-slate-100 space-y-6">
            <div className="p-5 bg-slate-50 rounded-3xl border border-slate-100">
              <div className="text-[10px] font-black text-slate-400 uppercase mb-4 tracking-widest">Role Switch / 演示角色切换</div>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { id: 'principal', label: 'Principal', labelCn: '校长' },
                  { id: 'teacher', label: 'Teacher', labelCn: '老师' },
                  { id: 'parent', label: 'Parent', labelCn: '家长' },
                  { id: 'student', label: 'Student', labelCn: '学生' },
                  { id: 'bureau', label: 'Bureau', labelCn: '教育局' },
                  { id: 'admin', label: 'Admin', labelCn: '管理员' },
                ].map((r) => (
                  <button
                    key={r.id}
                    onClick={() => switchRole(r.id as any)}
                    className={cn(
                      "px-2 py-2 rounded-xl text-[10px] font-bold transition-all flex flex-col items-center",
                      user.role === r.id ? "bg-blue-600 text-white shadow-lg shadow-blue-200" : "bg-white text-slate-600 border border-slate-200 hover:border-blue-300"
                    )}
                  >
                    <span>{r.label}</span>
                    <span className="opacity-60 font-medium">{r.labelCn}</span>
                  </button>
                ))}
              </div>
            </div>
            
            <div className="flex items-center gap-4 p-2">
              <div className="w-12 h-12 rounded-2xl bg-slate-100 flex items-center justify-center overflow-hidden border border-slate-200">
                <UserIcon className="w-7 h-7 text-slate-400" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-black text-slate-900 truncate">{user.displayName}</div>
                <div className="text-xs text-slate-400 truncate font-medium capitalize">{user.role}</div>
              </div>
            </div>
            
            <button onClick={handleLogout} className="w-full flex items-center gap-3 p-4 text-slate-500 hover:text-red-600 hover:bg-red-50 rounded-2xl transition-all font-bold text-sm">
              <LogOut className="w-5 h-5" />
              <div className="flex flex-col items-start">
                <span>Logout</span>
                <span className="text-[10px] opacity-60 font-medium">退出登录</span>
              </div>
            </button>
          </div>
        </aside>

        <main className="flex-1 p-10 overflow-y-auto">
          <AnimatePresence mode="wait">
            {activeTab === 'dashboard' && (
              <motion.div key="dashboard" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.18 }}>
                {user.role === 'principal' && <PrincipalDashboard />}
                {user.role === 'admin' && <PrincipalDashboard />}
                {user.role === 'teacher' && <TeacherDashboard />}
                {user.role === 'parent' && <ParentDashboard />}
                {user.role === 'bureau' && <BureauDashboard />}
                {user.role === 'doctor' && <PrincipalDashboard />}
                {user.role === 'student' && <StudentHome user={user} setActiveTab={setActiveTab} />}
              </motion.div>
            )}
            {activeTab === 'student-home' && (
              <motion.div key="student-home" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.18 }}>
                <StudentHome user={user} setActiveTab={setActiveTab} />
              </motion.div>
            )}
            {activeTab === 'student-checkin' && (
              <motion.div key="student-checkin" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.18 }}>
                <StudentCheckin user={user} />
              </motion.div>
            )}
            {activeTab === 'student-qa' && (
              <motion.div key="student-qa" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.18 }}>
                <StudentQA user={user} />
              </motion.div>
            )}
            {activeTab === 'student-points' && (
              <motion.div key="student-points" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.18 }}>
                <StudentPoints user={user} />
              </motion.div>
            )}
            {activeTab === 'student-profile' && (
              <motion.div key="student-profile" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.18 }}>
                <StudentProfile user={user} handleLogout={handleLogout} />
              </motion.div>
            )}
            {activeTab === 'monitor' && (
              <motion.div key="monitor" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.18 }}>
                <Monitor />
              </motion.div>
            )}
            {activeTab === 'agent' && (
              <motion.div key="agent" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.18 }}>
                <AgenticDashboard user={user} classrooms={classrooms} />
              </motion.div>
            )}
            {activeTab === 'parent' && (
              <motion.div key="parent" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.18 }}>
                <ParentPortal />
              </motion.div>
            )}
            {activeTab === 'alerts' && (
              <motion.div key="alerts" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.18 }}>
                <Alerts user={user} />
              </motion.div>
            )}
            {activeTab === 'health' && (
              <motion.div key="health" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.18 }}>
                <HealthGuide user={user} classrooms={classrooms} />
              </motion.div>
            )}
            {activeTab === 'management' && (
              <motion.div key="management" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.18 }}>
                <ManagementSystem classrooms={classrooms} setClassrooms={setClassrooms} students={students} setStudents={setStudents} totalFloors={totalFloors} setTotalFloors={setTotalFloors} />
              </motion.div>
            )}
            {activeTab === 'voiceprint' && (
              <motion.div key="voiceprint" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.18 }}>
                <VoiceprintEnrollment students={students} setStudents={setStudents} />
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
    </ErrorBoundary>
  );
}

function NavItem({ active, icon: Icon, label, subLabel, onClick, badge }: { active: boolean, icon: any, label: string, subLabel?: string, onClick: () => void, badge?: string }) {
  return (
    <button 
      onClick={onClick} 
      className={cn(
        "w-full flex items-center gap-4 p-4 rounded-2xl transition-all font-black text-sm relative group", 
        active 
          ? "bg-slate-900 text-white shadow-2xl shadow-slate-200" 
          : "text-slate-500 hover:bg-slate-50 hover:text-slate-900"
      )}
    >
      <Icon className={cn("w-5 h-5 transition-transform group-hover:scale-110", active ? "text-blue-500" : "text-slate-400 group-hover:text-slate-900")} />
      <div className="flex-1 text-left flex flex-col">
        <span>{label}</span>
        {subLabel && <span className="text-[10px] opacity-50 font-bold">{subLabel}</span>}
      </div>
      {badge && (
        <span className={cn(
          "px-2 py-0.5 rounded-lg text-[10px] font-black", 
          active ? "bg-blue-600 text-white" : "bg-red-500 text-white"
        )}>
          {badge}
        </span>
      )}
      {active && (
        <motion.div 
          layoutId="active-nav"
          className="absolute left-[-8px] w-1.5 h-8 bg-blue-600 rounded-full"
        />
      )}
    </button>
  );
}

// --- Student Components ---

function StudentHome({ user, setActiveTab }: { user: UserProfile, setActiveTab: (tab: any) => void }) {
  const riskScore = 45; // Mock risk score
  
  const getRiskContent = () => {
    if (riskScore < 40) return { color: 'text-emerald-600', bg: 'bg-emerald-50', border: 'border-emerald-100', icon: CheckCircle2, title: 'School is healthy today! \n 今天学校很健康！', desc: 'Remember to drink more water and keep up your good habits! \n 记得多喝水，保持好习惯～' };
    if (riskScore < 70) return { color: 'text-orange-600', bg: 'bg-orange-50', border: 'border-orange-100', icon: AlertTriangle, title: 'Medium Flu Risk \n 学校流感风险中等', desc: 'Wash your hands frequently and tell your teacher if you feel unwell. \n 今天记得勤洗手，如果感觉不舒服告诉老师哦' };
    return { color: 'text-red-600', bg: 'bg-red-50', border: 'border-red-100', icon: ShieldAlert, title: 'High Flu Risk \n 学校流感风险较高', desc: 'Be extra careful today! Wear a mask and tell your parents or teacher immediately if you feel sick. \n 今天要特别注意！戴好口罩，发现不舒服马上告诉家长和老师' };
  };

  const risk = getRiskContent();
  const tips = [
    "Flu viruses spread through droplets. Cover your mouth and nose with a tissue or your elbow when coughing or sneezing! \n 流感病毒主要通过飞沫传播，咳嗽或打喷嚏时用纸巾或肘部遮住口鼻，可以减少传播哦～",
    "Frequent handwashing is the simplest and most effective way to prevent flu. Scrub for at least 20 seconds! \n 勤洗手是预防流感最简单有效的方法，记得用肥皂搓手至少20秒！",
    "Keep the classroom ventilated. Fresh air leaves viruses with nowhere to hide. \n 保持教室通风，新鲜空气能让病毒无处藏身。",
    "Eat more fruits and vegetables. Plenty of sleep will keep your immunity strong! \n 多吃水果蔬菜，充足的睡眠能让你的免疫力棒棒哒！"
  ];
  // useMemo so the tip doesn't re-randomize on every parent re-render
  const randomTip = useMemo(() => tips[Math.floor(Math.random() * tips.length)], []);

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-2xl mx-auto space-y-8">
      {/* Profile Card */}
      <div className="bg-white p-8 rounded-[40px] border border-slate-100 shadow-xl flex items-center gap-6">
        <div className="text-6xl">{user.avatar || '🧒'}</div>
        <div className="flex-1">
          <h2 className="text-2xl font-black text-slate-900 leading-tight">
            Hello, {user.displayName?.split(' / ')[0]}!<br />
            <span className="text-sm text-slate-400 font-bold">你好，{user.displayName?.split(' / ')[0]} 同学！</span>
          </h2>
          <div className="text-slate-500 font-medium mt-1">{user.grade}{user.class} | ID: {user.studentId}</div>
          <div className="mt-3 inline-flex items-center gap-2 px-3 py-1 bg-yellow-50 text-yellow-700 rounded-full text-xs font-bold border border-yellow-100">
            <Trophy className="w-4 h-4" />
            Health Points: {user.healthPoints || 0} ⭐
          </div>
        </div>
      </div>

      {/* Risk Card */}
      <div className={cn("p-8 rounded-[40px] border shadow-sm flex gap-6 items-start", risk.bg, risk.border)}>
        <div className={cn("w-14 h-14 rounded-2xl flex items-center justify-center shrink-0", risk.color.replace('text', 'bg').replace('600', '500'))}>
          <risk.icon className="text-white w-8 h-8" />
        </div>
        <div className="space-y-2">
          <h3 className={cn("text-xl font-black flex flex-col", risk.color)}>
            <span>{risk.title.split(' \n ')[0]}</span>
            <span className="text-sm opacity-60 font-bold">{risk.title.split(' \n ')[1]}</span>
          </h3>
          <p className="text-slate-600 font-medium leading-relaxed flex flex-col">
            <span>{risk.desc.split(' \n ')[0]}</span>
            <span className="text-xs opacity-60 mt-1">{risk.desc.split(' \n ')[1]}</span>
          </p>
        </div>
      </div>

      <div className="bg-white p-8 rounded-[40px] border border-slate-100 shadow-sm flex items-center justify-between">
        <div className="space-y-1">
          <h4 className="font-bold text-slate-900 flex flex-col">
            <span>Daily Health Check-in</span>
            <span className="text-[10px] text-slate-400 font-bold">今日健康打卡</span>
          </h4>
          <p className="text-xs text-slate-500">Check in daily to earn points! \n 每天坚持打卡，领取健康积分</p>
        </div>
        <button 
          onClick={() => setActiveTab('student-checkin')}
          className="px-6 py-3 bg-blue-600 text-white rounded-2xl font-bold shadow-lg shadow-blue-100 hover:bg-blue-700 transition-all text-xs flex flex-col items-center"
        >
          <span>Check-in Now</span>
          <span className="text-[8px] opacity-60">立即打卡</span>
        </button>
      </div>

      {/* Health Tip */}
      <div className="bg-blue-600 p-8 rounded-[40px] text-white space-y-4 relative overflow-hidden shadow-2xl shadow-blue-200">
        <div className="absolute top-[-20px] right-[-20px] opacity-10">
          <Sparkles className="w-40 h-40" />
        </div>
        <div className="flex items-center gap-2 font-black text-sm uppercase tracking-widest opacity-80">
          <Zap className="w-4 h-4" />
          今日健康贴士 / Health Tip
        </div>
        <p className="text-xl font-bold leading-relaxed relative z-10 flex flex-col">
          <span>{randomTip.split(' \n ')[0]}</span>
          <span className="text-sm opacity-60 mt-2 font-medium">{randomTip.split(' \n ')[1]}</span>
        </p>
      </div>
    </motion.div>
  );
}

function StudentCheckin({ user }: { user: UserProfile }) {
  const [feeling, setFeeling] = useState<'good' | 'okay' | 'unwell' | 'sick' | null>(null);
  const [symptoms, setSymptoms] = useState<string[]>([]);
  const [temp, setTemp] = useState('');
  const [note, setNote] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isDone, setIsDone] = useState(false);

  const symptomList = [
    { en: 'Fever', zh: '发烧' },
    { en: 'Cough', zh: '咳嗽' },
    { en: 'Runny Nose', zh: '流鼻涕' },
    { en: 'Sore Throat', zh: '嗓子疼' },
    { en: 'Headache', zh: '头疼' },
    { en: 'Muscle Pain', zh: '肌肉酸痛' },
    { en: 'Fatigue', zh: '乏力' },
    { en: 'Other', zh: '其他' }
  ];

  const toggleSymptom = (s: string) => {
    setSymptoms(prev => prev.includes(s) ? prev.filter(item => item !== s) : [...prev, s]);
  };

  const handleSubmit = async () => {
    if (!feeling) return;
    setIsSubmitting(true);
    
    const checkin: HealthCheckin = {
      userId: user.uid,
      date: new Date().toISOString().split('T')[0],
      feeling,
      symptoms,
      temperature: temp ? parseFloat(temp) : null,
      note,
      createdAt: Timestamp.now()
    };

    try {
      const checkinId = `${user.uid}_${checkin.date}`;
      await setDoc(doc(db, 'healthCheckins', checkinId), checkin);
      
      // Update points
      const pointsToAdd = (feeling === 'unwell' || feeling === 'sick') ? 15 : 10;
      const newPoints = (user.healthPoints || 0) + pointsToAdd;
      await setDoc(doc(db, 'users', user.uid), { ...user, healthPoints: newPoints }, { merge: true });
      
      setIsDone(true);
    } catch (error) {
      handleFirestoreError(error, OperationType.WRITE, 'healthCheckins');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (isDone) {
    return (
      <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="max-w-md mx-auto text-center space-y-8 py-12">
        <div className="w-24 h-24 bg-emerald-100 text-emerald-600 rounded-[40px] flex items-center justify-center mx-auto shadow-xl shadow-emerald-50">
          <CheckCircle2 className="w-12 h-12" />
        </div>
        <div className="space-y-2">
          <h2 className="text-3xl font-black text-slate-900 flex flex-col">
            <span>Check-in Success!</span>
            <span className="text-sm text-slate-400 font-bold">打卡成功！</span>
          </h2>
          <p className="text-slate-500 font-medium flex flex-col">
            <span>Great job! You've earned health points.</span>
            <span className="text-xs opacity-60">你今天真棒，获得了积分奖励 ⭐</span>
          </p>
        </div>
        <div className="bg-emerald-50 p-6 rounded-3xl border border-emerald-100 text-emerald-700 font-bold">
          Health Points +{(feeling === 'unwell' || feeling === 'sick') ? 15 : 10}
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-2xl mx-auto space-y-10">
      <header className="text-center space-y-2">
        <h2 className="text-4xl font-black text-slate-900 tracking-tight">
          Health Check-in<br />
          <span className="text-xl text-slate-400 font-serif italic">健康打卡</span>
        </h2>
        <p className="text-slate-500 font-medium">Record your status to keep school safe. \n 记录你的健康状态，守护校园安全</p>
      </header>

      <div className="bg-white p-10 rounded-[40px] border border-slate-100 shadow-xl space-y-10">
        {/* Feeling Selection */}
        <div className="space-y-6">
          <label className="text-sm font-black text-slate-400 uppercase tracking-widest">How do you feel? / 今日感受</label>
          <div className="grid grid-cols-4 gap-4">
            {[
              { id: 'good', label: 'Good \n 很好', icon: Smile, color: 'text-emerald-500', bg: 'bg-emerald-50' },
              { id: 'okay', label: 'Okay \n 一般', icon: Meh, color: 'text-blue-500', bg: 'bg-blue-50' },
              { id: 'unwell', label: 'Unwell \n 不舒服', icon: Frown, color: 'text-orange-500', bg: 'bg-orange-50' },
              { id: 'sick', label: 'Sick \n 生病了', icon: Heart, color: 'text-red-500', bg: 'bg-red-50' },
            ].map((f) => (
              <button
                key={f.id}
                onClick={() => setFeeling(f.id as any)}
                className={cn(
                  "flex flex-col items-center gap-3 p-4 rounded-3xl border-2 transition-all",
                  feeling === f.id ? "border-blue-600 bg-blue-50/50 scale-105" : "border-slate-50 hover:border-slate-200"
                )}
              >
                <f.icon className={cn("w-10 h-10", f.color)} />
                <span className="text-[10px] font-bold text-slate-700 whitespace-pre-wrap">{f.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Symptoms */}
        <AnimatePresence>
          {(feeling === 'unwell' || feeling === 'sick') && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} className="space-y-6 overflow-hidden">
              <label className="text-sm font-black text-slate-400 uppercase tracking-widest">Symptoms / 具体症状</label>
              <div className="grid grid-cols-3 gap-3">
                {symptomList.map(s => (
                  <button
                    key={s.en}
                    onClick={() => toggleSymptom(s.en)}
                    className={cn(
                      "px-4 py-3 rounded-2xl border text-[10px] font-bold transition-all flex flex-col items-center",
                      symptoms.includes(s.en) ? "bg-blue-600 text-white border-blue-600" : "bg-slate-50 text-slate-600 border-slate-100 hover:border-blue-200"
                    )}
                  >
                    <span>{s.en}</span>
                    <span className="opacity-60">{s.zh}</span>
                  </button>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Temperature */}
        <div className="space-y-4">
          <label className="text-sm font-black text-slate-400 uppercase tracking-widest">Temperature (℃) / 今日体温</label>
          <div className="relative">
            <ThermometerIcon className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 w-5 h-5" />
            <input 
              type="number" 
              step="0.1"
              value={temp}
              onChange={(e) => setTemp(e.target.value)}
              placeholder="36.5"
              className="w-full pl-12 pr-4 py-4 bg-slate-50 border-none rounded-2xl focus:ring-2 focus:ring-blue-600 outline-none transition-all"
            />
          </div>
        </div>

        {/* Note */}
        <div className="space-y-4">
          <label className="text-sm font-black text-slate-400 uppercase tracking-widest">Note / 想说的话</label>
          <textarea 
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="How are you feeling today? Anything to tell your teacher? \n 今天感觉怎么样？有什么想告诉老师的吗？"
            maxLength={100}
            className="w-full p-6 bg-slate-50 border-none rounded-[32px] focus:ring-2 focus:ring-blue-600 outline-none transition-all h-32 resize-none"
          />
        </div>

        <button 
          onClick={handleSubmit}
          disabled={!feeling || isSubmitting}
          className="w-full py-5 bg-blue-600 text-white rounded-[28px] font-black text-xl shadow-2xl shadow-blue-200 hover:bg-blue-700 hover:scale-[1.02] active:scale-[0.98] transition-all disabled:opacity-50 flex flex-col items-center justify-center"
        >
          <div className="flex items-center gap-3">
            {isSubmitting ? <Loader2 className="w-6 h-6 animate-spin" /> : <CheckCircle2 className="w-6 h-6" />}
            <span>Submit Check-in</span>
          </div>
          <span className="text-xs opacity-60 font-bold mt-1">完成打卡</span>
        </button>
      </div>
      
      <p className="text-center text-[10px] text-slate-400 px-10">
        Privacy: Your data is only for school flu prevention. We protect your privacy. \n 隐私保护：你的打卡数据仅供学校流感防控使用，我们承诺保护你的个人隐私。
      </p>
    </motion.div>
  );
}

function StudentQA({ user }: { user: UserProfile }) {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'assistant', content: 'Hello! I am "Xiao Wei", your health assistant 🤖. Do you have any questions about flu prevention or your health? \n 你好呀！我是你的健康助手“小卫”🤖。关于流感预防或者身体健康，你有什么想问我的吗？' }
  ]);
  const [isTyping, setIsTyping] = useState(false);

  const quickQuestions = [
    'What is flu? \n 流感是什么？', 
    'How to prevent cold? \n 怎么预防感冒？', 
    'What if I have a fever? \n 发烧了怎么办？', 
    'When to go to hospital? \n 什么时候要去医院？', 
    'How to wear a mask? \n 口罩怎么戴？', 
    'How to wash hands? \n 手怎么洗干净？'
  ];

  const handleSend = async (text?: string) => {
    const messageText = text || query;
    if (!messageText.trim()) return;
    
    const userMsg = { role: 'user' as const, content: messageText };
    setMessages(prev => [...prev, userMsg]);
    setQuery('');
    setIsTyping(true);

    try {
      const systemPrompt = SYSTEM_PROMPTS.STUDENT + "\n\n" + SYSTEM_PROMPTS.SAFETY;
      const result = await callFluGuardAI({
        role: 'student',
        message: messageText,
        system_prompt: systemPrompt,
      });
      setMessages(prev => [...prev, { role: 'assistant', content: result.text || 'Sorry, Xiao Wei was distracted just now. Can you say that again? \n 抱歉，小卫刚才走神了，能再说一遍吗？' }]);
    } catch (error) {
      console.error('QA failed:', error);
      setMessages(prev => [...prev, { role: 'assistant', content: 'Cannot reach FluGuard AI backend. Please ensure the backend is running on port 8000. \n 无法连接到流感卫士AI后端，请确认后端服务运行在8000端口。' }]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto h-[calc(100vh-160px)] flex flex-col space-y-6">
      <header className="flex items-center gap-4 shrink-0">
        <div className="w-12 h-12 bg-blue-600 rounded-2xl flex items-center justify-center text-white shadow-lg">
          <BrainCircuit className="w-6 h-6" />
        </div>
        <div>
          <h2 className="text-2xl font-black text-slate-900 flex flex-col">
            <span>Health Q&A</span>
            <span className="text-sm text-slate-400 font-bold">健康问答</span>
          </h2>
          <p className="text-xs text-slate-500 font-bold uppercase tracking-widest">Chat with Xiao Wei / 与小卫对话</p>
        </div>
      </header>

      <div className="flex-1 bg-white rounded-[40px] border border-slate-100 shadow-xl flex flex-col overflow-hidden">
        {/* Quick Questions */}
        <div className="p-4 border-b border-slate-50 flex gap-2 overflow-x-auto no-scrollbar shrink-0">
          {quickQuestions.map(q => (
            <button 
              key={q} 
              onClick={() => handleSend(q)}
              className="whitespace-nowrap px-4 py-2 bg-slate-50 hover:bg-blue-50 hover:text-blue-600 rounded-full text-xs font-bold text-slate-500 transition-all border border-slate-100"
            >
              {q}
            </button>
          ))}
        </div>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto p-8 space-y-6">
          {messages.map((msg, i) => (
            <motion.div 
              key={i}
              initial={{ opacity: 0, y: 10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              className={cn(
                "flex gap-4 max-w-[85%]",
                msg.role === 'user' ? "ml-auto flex-row-reverse" : "mr-auto"
              )}
            >
              <div className={cn(
                "w-10 h-10 rounded-2xl flex items-center justify-center shrink-0 shadow-lg text-xl",
                msg.role === 'user' ? "bg-blue-600 text-white" : "bg-slate-100 text-slate-600"
              )}>
                {msg.role === 'user' ? <UserIcon className="w-5 h-5" /> : '🤖'}
              </div>
              <div className={cn(
                "p-5 rounded-[28px] text-sm leading-relaxed font-medium shadow-sm border whitespace-pre-wrap",
                msg.role === 'user' 
                  ? "bg-blue-600 text-white border-blue-500 rounded-tr-none" 
                  : "bg-slate-50 text-slate-700 border-slate-100 rounded-tl-none"
              )}>
                {msg.content}
              </div>
            </motion.div>
          ))}
          {isTyping && (
            <div className="flex gap-4 mr-auto">
              <div className="w-10 h-10 rounded-2xl bg-slate-100 flex items-center justify-center shrink-0 shadow-lg text-xl">🤖</div>
              <div className="p-5 bg-slate-50 rounded-[28px] rounded-tl-none border border-slate-100 flex gap-1.5 items-center">
                <motion.div animate={{ scale: [1, 1.2, 1] }} transition={{ repeat: Infinity, duration: 0.6 }} className="w-1.5 h-1.5 bg-blue-400 rounded-full" />
                <motion.div animate={{ scale: [1, 1.2, 1] }} transition={{ repeat: Infinity, duration: 0.6, delay: 0.2 }} className="w-1.5 h-1.5 bg-blue-400 rounded-full" />
                <motion.div animate={{ scale: [1, 1.2, 1] }} transition={{ repeat: Infinity, duration: 0.6, delay: 0.4 }} className="w-1.5 h-1.5 bg-blue-400 rounded-full" />
              </div>
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="p-6 bg-slate-50 border-t border-slate-100">
          <div className="flex gap-4">
            <input 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Ask Xiao Wei... / 问问小卫健康知识..."
              className="flex-1 bg-white border-none rounded-2xl px-6 py-4 text-sm shadow-sm focus:ring-2 focus:ring-blue-600 transition-all outline-none"
            />
            <button 
              onClick={() => handleSend()}
              className="w-14 h-14 bg-blue-600 text-white rounded-2xl flex items-center justify-center shadow-lg shadow-blue-100 hover:bg-blue-700 transition-all"
            >
              <Send className="w-6 h-6" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function StudentPoints({ user }: { user: UserProfile }) {
  const leaderboard = [
    { name: 'Zhang T.', points: 980, rank: 1 },
    { name: 'Li T.', points: 920, rank: 2 },
    { name: 'Wang T.', points: 875, rank: 3 },
    { name: 'Zhao T.', points: 810, rank: 4 },
    { name: 'Sun T.', points: 790, rank: 5 },
    { name: 'Zhou T.', points: 750, rank: 6 },
    { name: 'Wu T.', points: 720, rank: 7 },
    { name: 'Zheng T.', points: 680, rank: 8 },
    { name: 'Feng T.', points: 650, rank: 9 },
    { name: 'Chen T.', points: 620, rank: 10 },
  ];

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-4xl mx-auto space-y-8">
      <header className="flex items-center justify-between">
        <h2 className="text-3xl font-black text-slate-900 flex flex-col">
          <span>Health Points</span>
          <span className="text-sm text-slate-400 font-bold">健康积分</span>
        </h2>
        <div className="bg-blue-600 text-white px-6 py-3 rounded-2xl font-black flex items-center gap-2 shadow-xl shadow-blue-100">
          <Trophy className="w-5 h-5" />
          {user.healthPoints || 0} ⭐
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Rules Card */}
        <div className="bg-white p-8 rounded-[40px] border border-slate-100 shadow-sm space-y-6">
          <h3 className="text-xl font-bold flex items-center gap-2">
            <CheckSquare className="w-5 h-5 text-blue-600" />
            Points Rules / 积分规则
          </h3>
          <div className="space-y-4">
            {[
              { en: 'Daily Health Check-in', zh: '每日健康打卡', points: '+10', icon: CheckCircle2 },
              { en: 'Truthful Symptom Report', zh: '如实上报症状', points: '+15', icon: Sparkles },
              { en: '7-Day Streak', zh: '连续打卡7天', points: '+50', icon: Zap },
              { en: '30-Day Streak', zh: '连续打卡30天', points: '+200', icon: Trophy },
            ].map((rule, i) => (
              <div key={i} className="flex items-center justify-between p-4 bg-slate-50 rounded-2xl">
                <div className="flex items-center gap-3">
                  <rule.icon className="w-5 h-5 text-blue-500" />
                  <div className="flex flex-col">
                    <span className="font-bold text-slate-700 text-sm">{rule.en}</span>
                    <span className="text-[10px] text-slate-400 font-medium">{rule.zh}</span>
                  </div>
                </div>
                <span className="text-blue-600 font-black">{rule.points}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Stats Card */}
        <div className="bg-slate-900 p-8 rounded-[40px] text-white space-y-8 shadow-2xl">
          <h3 className="text-xl font-bold">My Achievements / 我的成就</h3>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-1">
              <div className="text-slate-400 text-[10px] font-bold uppercase">This Week / 本周打卡</div>
              <div className="text-3xl font-black">5/7 <span className="text-sm font-medium opacity-50">Days</span></div>
            </div>
            <div className="space-y-1">
              <div className="text-slate-400 text-[10px] font-bold uppercase">Streak / 连续打卡</div>
              <div className="text-3xl font-black">12 <span className="text-sm font-medium opacity-50">Days 🔥</span></div>
            </div>
            <div className="space-y-1">
              <div className="text-slate-400 text-[10px] font-bold uppercase">Rank / 本月排名</div>
              <div className="text-3xl font-black">9th <span className="text-sm font-medium opacity-50">名</span></div>
            </div>
            <div className="space-y-1">
              <div className="text-slate-400 text-[10px] font-bold uppercase">Total / 累计积分</div>
              <div className="text-3xl font-black">{user.healthPoints || 0} <span className="text-sm font-medium opacity-50">⭐</span></div>
            </div>
          </div>
          <div className="p-4 bg-white/10 rounded-2xl border border-white/10 text-xs text-slate-300">
            Keep going! Just 2 more days to get the "7-Day Streak" bonus! \n 加油！再坚持2天就能获得“连续打卡7天”额外奖励啦！
          </div>
        </div>
      </div>

      <div className="bg-white p-8 rounded-[40px] border border-slate-100 shadow-xl space-y-6">
        <h3 className="text-xl font-bold flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-blue-600" />
          <div className="flex flex-col">
            <span>Class Leaderboard</span>
            <span className="text-xs text-slate-400 font-bold">班级排行榜</span>
          </div>
        </h3>
        <div className="space-y-2">
          {leaderboard.map((item) => (
            <div key={item.rank} className={cn(
              "flex items-center justify-between p-4 rounded-2xl transition-all",
              item.name.includes(user.displayName?.split(' / ')[0] || '') ? "bg-blue-50 border border-blue-100" : "hover:bg-slate-50"
            )}>
              <div className="flex items-center gap-4">
                <div className={cn(
                  "w-8 h-8 rounded-lg flex items-center justify-center font-black text-sm",
                  item.rank === 1 ? "bg-yellow-100 text-yellow-700" : 
                  item.rank === 2 ? "bg-slate-100 text-slate-600" : 
                  item.rank === 3 ? "bg-orange-100 text-orange-700" : "text-slate-400"
                )}>
                  {item.rank === 1 ? '🥇' : item.rank === 2 ? '🥈' : item.rank === 3 ? '🥉' : item.rank}
                </div>
                <span className="font-bold text-slate-700">{item.name}</span>
              </div>
              <span className="font-black text-slate-900">{item.points} 分</span>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}

function StudentProfile({ user, handleLogout }: { user: UserProfile, handleLogout: () => void }) {
  const [avatar, setAvatar] = useState(user.avatar || '🧒');
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);
  
  const emojis = ['🧒', '👧', '👦', '👶', '🐱', '🐶', '🦊', '🐼', '🐨', '🦁', '🐯', '🐸', '🦄', '🦖', '🚀', '🎨', '⚽', '🎮', '🍦', '⭐'];

  const updateAvatar = async (e: string) => {
    setAvatar(e);
    setShowEmojiPicker(false);
    try {
      await setDoc(doc(db, 'users', user.uid), { ...user, avatar: e }, { merge: true });
    } catch (error) {
      console.error('Update avatar failed:', error);
    }
  };

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-2xl mx-auto space-y-8">
      <div className="bg-white p-10 rounded-[40px] border border-slate-100 shadow-xl space-y-10 text-center">
        <div className="relative inline-block">
          <div className="text-8xl p-6 bg-slate-50 rounded-[48px] border-4 border-white shadow-inner">
            {avatar}
          </div>
          <button 
            onClick={() => setShowEmojiPicker(!showEmojiPicker)}
            className="absolute bottom-0 right-0 w-12 h-12 bg-blue-600 text-white rounded-2xl flex items-center justify-center shadow-lg hover:scale-110 transition-all"
          >
            <Plus className="w-6 h-6" />
          </button>
          
          <AnimatePresence>
            {showEmojiPicker && (
              <motion.div 
                initial={{ opacity: 0, scale: 0.9, y: 10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.9, y: 10 }}
                className="absolute top-full mt-4 left-1/2 -translate-x-1/2 bg-white p-6 rounded-[32px] shadow-2xl border border-slate-100 z-50 w-64 grid grid-cols-5 gap-3"
              >
                {emojis.map(e => (
                  <button key={e} onClick={() => updateAvatar(e)} className="text-2xl hover:scale-125 transition-all">{e}</button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="space-y-2">
          <h2 className="text-3xl font-black text-slate-900">{user.displayName?.split(' / ')[0]}</h2>
          <p className="text-slate-500 font-medium">{user.grade}{user.class} | ID: {user.studentId}</p>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div className="p-6 bg-slate-50 rounded-3xl space-y-1">
            <div className="text-2xl font-black text-slate-900">28</div>
            <div className="text-[10px] font-bold text-slate-400 uppercase flex flex-col">
              <span>Check-ins</span>
              <span className="opacity-60">打卡次数</span>
            </div>
          </div>
          <div className="p-6 bg-slate-50 rounded-3xl space-y-1">
            <div className="text-2xl font-black text-emerald-600">Good</div>
            <div className="text-[10px] font-bold text-slate-400 uppercase flex flex-col">
              <span>Health Score</span>
              <span className="opacity-60">健康评分</span>
            </div>
          </div>
          <div className="p-6 bg-slate-50 rounded-3xl space-y-1">
            <div className="text-2xl font-black text-blue-600">{user.healthPoints}</div>
            <div className="text-[10px] font-bold text-slate-400 uppercase flex flex-col">
              <span>Total Points</span>
              <span className="opacity-60">累计积分</span>
            </div>
          </div>
        </div>

        <div className="space-y-4 pt-6">
          <button onClick={handleLogout} className="w-full py-5 bg-red-50 text-red-600 rounded-[28px] font-black text-lg hover:bg-red-100 transition-all flex flex-col items-center justify-center">
            <span>Logout</span>
            <span className="text-xs opacity-60 font-bold">退出登录</span>
          </button>
        </div>
      </div>
    </motion.div>
  );
}

function PrincipalDashboard() {
  return (
    <div className="space-y-12 max-w-[1600px] mx-auto">
      <header className="flex flex-col lg:flex-row lg:items-end justify-between gap-6">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-blue-600 font-black text-xs uppercase tracking-widest">
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" />
            Live System Status
          </div>
          <h2 className="text-5xl font-black text-slate-900 tracking-tight">
            School Health Overview<br />
            <span className="text-slate-400 font-serif italic text-4xl">全校健康概览</span>
          </h2>
          <p className="text-slate-500 font-medium">Shenyang Experimental Primary School · April 7, 2026<br/><span className="text-xs opacity-60">沈阳市实验小学</span></p>
        </div>
        <div className="flex gap-4">
          <div className="px-6 py-3 bg-emerald-50 text-emerald-600 rounded-2xl text-sm font-black flex items-center gap-3 border border-emerald-100">
            <CheckCircle2 className="w-5 h-5" /> 
            <div className="flex flex-col items-start">
              <span>Disinfection Completed</span>
              <span className="text-[10px] opacity-60 font-medium">校园消杀已完成</span>
            </div>
          </div>
          <button className="p-3 bg-white border border-slate-200 rounded-2xl hover:bg-slate-50 transition-colors">
            <Search className="w-6 h-6 text-slate-400" />
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
        <StatCard label="Total Coughs" subLabel="全校咳嗽总数" value="1,284" trend="+12%" icon={Activity} color="blue" />
        <StatCard label="Avg Risk Index" subLabel="平均风险指数" value="68" trend="+5%" icon={ShieldAlert} color="orange" />
        <StatCard label="Abnormal Classes" subLabel="异常班级数" value="3" trend="+1" icon={AlertTriangle} color="red" />
        <StatCard label="Disinfection Rate" subLabel="消杀覆盖率" value="100%" trend="Stable" icon={CheckCircle2} color="green" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
        <div className="lg:col-span-2 bento-card space-y-8">
          <div className="flex items-center justify-between">
            <h3 className="text-2xl font-black flex items-center gap-3">
              <TrendingUp className="w-6 h-6 text-blue-600" /> 
              <div className="flex flex-col">
                <span>School Epidemic Trend (24h)</span>
                <span className="text-sm opacity-50">全校流行趋势</span>
              </div>
            </h3>
            <div className="flex gap-2">
              <button className="px-3 py-1 bg-slate-100 text-slate-600 rounded-lg text-[10px] font-black">24H</button>
              <button className="px-3 py-1 bg-white text-slate-400 rounded-lg text-[10px] font-black">7D</button>
            </div>
          </div>
          <div className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={MOCK_TRENDS}>
                <defs>
                  <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#2563eb" stopOpacity={0.15}/>
                    <stop offset="95%" stopColor="#2563eb" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 12, fontWeight: 600}} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 12, fontWeight: 600}} />
                <Tooltip 
                  contentStyle={{ borderRadius: '24px', border: 'none', boxShadow: '0 20px 25px -5px rgba(0,0,0,0.1)', padding: '16px' }}
                />
                <Area type="monotone" dataKey="count" stroke="#2563eb" strokeWidth={4} fillOpacity={1} fill="url(#colorCount)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bento-card space-y-8">
          <h3 className="text-2xl font-black flex items-center gap-3 text-red-600">
            <AlertTriangle className="w-6 h-6" /> 
            <div className="flex flex-col">
              <span>High Risk Classes</span>
              <span className="text-sm opacity-50">高风险班级</span>
            </div>
          </h3>
          <div className="space-y-6">
            {MOCK_CLASSROOMS.filter(c => c.fluRiskLevel === 'high' || c.fluRiskLevel === 'medium').map(room => (
              <div key={room.id} className="p-6 bg-slate-50 rounded-3xl flex items-center justify-between group hover:bg-slate-100 transition-all cursor-pointer border border-transparent hover:border-slate-200">
                <div className="space-y-1">
                  <div className="font-black text-slate-900 text-lg">{room.name}</div>
                  <div className="text-xs font-bold text-slate-400 uppercase tracking-widest">24h Cough: {room.coughCount24h}</div>
                </div>
                <div className={cn(
                  "px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest",
                  room.fluRiskLevel === 'high' ? "bg-red-600 text-white shadow-lg shadow-red-100" : "bg-orange-500 text-white shadow-lg shadow-orange-100"
                )}>
                  {room.fluRiskLevel === 'high' ? 'High' : 'Medium'}
                </div>
              </div>
            ))}
          </div>
          <button className="w-full py-4 bg-slate-900 text-white rounded-2xl font-black text-sm hover:bg-slate-800 transition-colors">
            View All Classes / 查看全部班级
          </button>
        </div>
      </div>
    </div>
  );
}

function TeacherDashboard() {
  return (
    <div className="space-y-12 max-w-[1600px] mx-auto">
      <header className="flex flex-col lg:flex-row lg:items-end justify-between gap-6">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-red-600 font-black text-xs uppercase tracking-widest">
            <div className="w-2 h-2 bg-red-600 rounded-full animate-pulse" />
            Class Health Alert
          </div>
          <h2 className="text-5xl font-black text-slate-900 tracking-tight">
            Class Health Management<br />
            <span className="text-slate-400 font-serif italic text-4xl">班级健康管理</span>
          </h2>
          <p className="text-slate-500 font-medium">Grade 1 Class 1 · April 7, 2026<br/><span className="text-xs opacity-60">一年级一班</span></p>
        </div>
        <div className="flex gap-4">
          <div className="px-6 py-3 bg-red-50 text-red-600 rounded-2xl text-sm font-black flex items-center gap-3 border border-red-100">
            <AlertTriangle className="w-5 h-5" /> 
            <div className="flex flex-col items-start">
              <span>Class Flu Alert</span>
              <span className="text-[10px] opacity-60 font-medium">班级流感预警中</span>
            </div>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
        <StatCard label="Today's Coughs" subLabel="今日咳嗽次数" value="156" trend="+45%" icon={Activity} color="red" />
        <StatCard label="Check-in Rate" subLabel="今日打卡率" value="38/45" trend="84%" icon={CheckSquare} color="blue" />
        <StatCard label="Unwell Students" subLabel="身体不适学生" value="2" trend="Reported" icon={AlertTriangle} color="orange" />
        <StatCard label="CO2 Level" subLabel="CO2 浓度" value="850ppm" trend="Good" icon={Wind} color="green" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
        <div className="lg:col-span-2 space-y-10">
          <div className="bento-card space-y-8">
            <h3 className="text-2xl font-black flex items-center gap-3">
              <CheckSquare className="w-6 h-6 text-blue-600" /> 
              <div className="flex flex-col">
                <span>Health Check-in Summary</span>
                <span className="text-sm opacity-50">今日健康打卡汇总</span>
              </div>
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-6 bg-orange-50 rounded-3xl border border-orange-100 flex items-center gap-4">
                <div className="w-12 h-12 bg-orange-100 rounded-2xl flex items-center justify-center text-orange-600">
                  <Frown className="w-6 h-6" />
                </div>
                <div>
                  <div className="text-sm font-black text-orange-900">Zhang Xiaoming / 张小明</div>
                  <div className="text-xs text-orange-700 font-medium">Feeling: Unwell (Cough, Fever)</div>
                </div>
              </div>
              <div className="p-6 bg-red-50 rounded-3xl border border-red-100 flex items-center gap-4">
                <div className="w-12 h-12 bg-red-100 rounded-2xl flex items-center justify-center text-red-600">
                  <Heart className="w-6 h-6" />
                </div>
                <div>
                  <div className="text-sm font-black text-red-900">Li Hua / 李华</div>
                  <div className="text-xs text-red-700 font-medium">Feeling: Sick (Fever, Headache)</div>
                </div>
              </div>
            </div>
          </div>

          <div className="bento-card space-y-8">
            <h3 className="text-2xl font-black flex items-center gap-3">
              <Users className="w-6 h-6 text-blue-600" /> 
              <div className="flex flex-col">
                <span>Student Health List</span>
                <span className="text-sm opacity-50">学生健康状态名单</span>
              </div>
            </h3>
          <div className="overflow-hidden rounded-3xl border border-slate-100">
            <table className="w-full text-left">
              <thead className="bg-slate-50 text-slate-400 text-[10px] font-black uppercase tracking-widest">
                <tr>
                  <th className="px-8 py-6">Name / 姓名</th>
                  <th className="px-8 py-6">Status / 状态</th>
                  <th className="px-8 py-6">Cough / 咳嗽</th>
                  <th className="px-8 py-6">Last / 检测</th>
                  <th className="px-8 py-6">Advice / 建议</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {MOCK_STUDENTS.map(student => (
                  <tr key={student.id} className="hover:bg-slate-50 transition-colors group">
                    <td className="px-8 py-6">
                      <div className="font-black text-slate-900">{student.name}</div>
                      <div className="text-[10px] font-bold text-slate-400 uppercase">ID: {student.id}</div>
                    </td>
                    <td className="px-8 py-6">
                      <span className={cn(
                        "px-3 py-1.5 rounded-xl text-[10px] font-black uppercase tracking-widest",
                        student.absenceStatus === 'present' ? "bg-emerald-50 text-emerald-600" : "bg-red-50 text-red-600"
                      )}>
                        {student.absenceStatus === 'present' ? 'Present / 在校' : 'Absent / 缺勤'}
                      </span>
                    </td>
                    <td className="px-8 py-6">
                      <div className={cn("font-black", student.coughCount > 20 ? "text-red-600" : "text-slate-900")}>
                        {student.coughCount} Times
                      </div>
                    </td>
                    <td className="px-8 py-6 text-sm font-medium text-slate-500">{student.lastCough}</td>
                    <td className="px-8 py-6">
                      <div className="text-xs font-bold text-slate-600 bg-slate-100 px-3 py-1.5 rounded-lg inline-block">
                        {student.coughCount > 20 ? 'Home Observation / 建议居家观察' : 'Continued Monitor / 持续监测'}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="space-y-10">
          <div className="bento-card space-y-6">
            <h3 className="text-xl font-black flex items-center gap-3">
              <Wind className="w-6 h-6 text-emerald-500" /> 
              <div className="flex flex-col">
                <span>Environment</span>
                <span className="text-sm opacity-50">环境监测</span>
              </div>
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-slate-50 rounded-2xl space-y-1">
                <div className="text-[10px] font-bold text-slate-400 uppercase">Temperature</div>
                <div className="text-xl font-black text-slate-900">22.5°C</div>
              </div>
              <div className="p-4 bg-slate-50 rounded-2xl space-y-1">
                <div className="text-[10px] font-bold text-slate-400 uppercase">Humidity</div>
                <div className="text-xl font-black text-slate-900">45%</div>
              </div>
            </div>
            <div className="p-4 bg-emerald-50 border border-emerald-100 rounded-2xl flex items-center gap-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-600" />
              <div className="text-xs font-bold text-emerald-700">Ventilation Good / 通风良好</div>
            </div>
          </div>

          <div className="bento-card bg-slate-900 text-white space-y-6">
            <h3 className="text-xl font-black flex items-center gap-3">
              <BrainCircuit className="w-6 h-6 text-blue-400" /> 
              <div className="flex flex-col">
                <span>AI Advice</span>
                <span className="text-sm opacity-50">AI 助手建议</span>
              </div>
            </h3>
            <p className="text-sm text-slate-400 leading-relaxed">
              Cough frequency increased by 15% in the last 2 hours. Increase ventilation and remind students to wear masks.
              <span className="block text-xs mt-2 opacity-60">检测到班级内咳嗽频率在过去 2 小时内上升了 15%，建议增加通风并提醒佩戴口罩。</span>
            </p>
            <button className="w-full py-3 bg-blue-600 rounded-xl font-black text-sm hover:bg-blue-700 transition-colors">
              Execute Disinfection / 执行消杀指令
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function ParentDashboard() {
  return (
    <div className="max-w-2xl mx-auto space-y-10">
      <header className="text-center space-y-3">
        <div className="w-24 h-24 bg-blue-600 rounded-[32px] flex items-center justify-center mx-auto shadow-2xl shadow-blue-200 transform -rotate-6">
          <UserIcon className="text-white w-12 h-12" />
        </div>
        <h2 className="text-4xl font-black text-slate-900 tracking-tight">
          Child Health Dashboard<br />
          <span className="text-slate-400 font-serif italic text-3xl">孩子健康看板</span>
        </h2>
        <p className="text-slate-500 font-medium">Zhang Xiaoming · Grade 1 Class 1<br/><span className="text-xs opacity-60">张小明 · 一年级一班</span></p>
      </header>

      <div className="bento-card space-y-10 p-10">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <div className="text-xs font-black text-slate-400 uppercase tracking-widest">Today's Status / 今日状态</div>
            <div className="text-3xl font-black text-slate-900">Mild Cough / 轻微咳嗽</div>
          </div>
          <div className="text-right space-y-1">
            <div className="text-xs font-black text-slate-400 uppercase tracking-widest">Check-in / 今日打卡</div>
            <div className="flex items-center gap-2 text-emerald-600 font-black text-xl">
              <CheckCircle2 className="w-5 h-5" />
              Completed / 已完成
            </div>
          </div>
        </div>

        <div className="p-6 bg-emerald-50 rounded-3xl border border-emerald-100 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="text-3xl">😷</div>
            <div>
              <div className="font-black text-emerald-900">Feeling: Unwell / 感觉不舒服</div>
              <div className="text-xs text-emerald-700 font-medium">Symptoms: Cough, Runny Nose / 症状：咳嗽、流鼻涕</div>
            </div>
          </div>
          <div className="text-right">
            <div className="text-[10px] font-black text-emerald-600 uppercase">Points Earned</div>
            <div className="text-xl font-black text-emerald-700">+15 ⭐</div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div className="p-6 bg-slate-50 rounded-3xl border border-slate-100 space-y-2">
            <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Class Risk / 班级风险</div>
            <div className="text-3xl font-black text-red-500">82 <span className="text-sm opacity-50">/ 100</span></div>
          </div>
          <div className="p-6 bg-slate-50 rounded-3xl border border-slate-100 space-y-2">
            <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest">School Risk / 全校风险</div>
            <div className="text-3xl font-black text-orange-500">65 <span className="text-sm opacity-50">/ 100</span></div>
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="text-lg font-black flex items-center gap-3">
            <BrainCircuit className="w-6 h-6 text-blue-600" /> 
            <div className="flex flex-col">
              <span>Gemma AI Advice</span>
              <span className="text-sm opacity-50">AI 贴心建议</span>
            </div>
          </h3>
          <div className="p-6 bg-blue-50 rounded-3xl border border-blue-100 text-slate-700 leading-relaxed relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4 opacity-10">
              <BrainCircuit className="w-20 h-20" />
            </div>
            <p className="relative z-10 font-medium">
              "Xiaoming coughed 12 times today, mostly in the morning. It's getting cold in Shenyang. Please keep him warm and observe tonight."
              <span className="block text-xs mt-2 opacity-60">“咱家小明今天在学校咳嗽了12次，主要是上午。沈阳现在降温，晚上回家给孩子熬点梨水，明天记得带个厚外套。”</span>
            </p>
          </div>
        </div>

        <button className="w-full py-5 bg-slate-900 text-white rounded-[24px] font-black text-lg flex items-center justify-center gap-3 hover:bg-slate-800 transition-all shadow-xl shadow-slate-200">
          <MessageSquare className="w-6 h-6" /> 
          <div className="flex flex-col items-center">
            <span>Chat with AI Doctor</span>
            <span className="text-[10px] opacity-60">咨询 AI 校医</span>
          </div>
        </button>
      </div>

      <div className="bento-card space-y-6">
        <h3 className="text-xl font-black mb-6">Class Dynamics / 班级健康动态</h3>
        <div className="space-y-6">
          <div className="flex items-start gap-4">
            <div className="w-3 h-3 rounded-full bg-red-500 mt-1.5 shrink-0 shadow-lg shadow-red-100" />
            <div className="text-sm font-medium text-slate-600 leading-relaxed">
              2 more absences today. Please take precautions.
              <span className="block text-xs opacity-60">班级今日新增 2 名缺勤学生，建议加强预防。</span>
            </div>
          </div>
          <div className="flex items-start gap-4">
            <div className="w-3 h-3 rounded-full bg-emerald-500 mt-1.5 shrink-0 shadow-lg shadow-emerald-100" />
            <div className="text-sm font-medium text-slate-600 leading-relaxed">
              Disinfection completed at 14:00.
              <span className="block text-xs opacity-60">学校已于 14:00 完成全校紫外线消杀。</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function BureauDashboard() {
  return (
    <div className="space-y-12 max-w-[1600px] mx-auto">
      <header className="flex flex-col lg:flex-row lg:items-end justify-between gap-6">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-blue-600 font-black text-xs uppercase tracking-widest">
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" />
            Regional Monitoring Active
          </div>
          <h2 className="text-5xl font-black text-slate-900 tracking-tight">
            Regional Monitoring Center<br />
            <span className="text-slate-400 font-serif italic text-4xl">区域流感监测中心</span>
          </h2>
          <p className="text-slate-500 font-medium">Shenyang Education Bureau · Live Dashboard<br/><span className="text-xs opacity-60">沈阳市教育局 · 实时监测看板</span></p>
        </div>
        <div className="flex gap-4">
          <div className="px-6 py-3 bg-blue-50 text-blue-600 rounded-2xl text-sm font-black flex items-center gap-3 border border-blue-100">
            <Building2 className="w-5 h-5" /> 
            <div className="flex flex-col items-start">
              <span>12 Districts Connected</span>
              <span className="text-[10px] opacity-60 font-medium">全市 12 个区域已联网</span>
            </div>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
        <StatCard label="Active Subjects" subLabel="全市活跃监测对象" value="5,420" trend="+15%" icon={Activity} color="red" />
        <StatCard label="Avg Regional Risk" subLabel="区域平均风险" value="62" trend="+8%" icon={ShieldAlert} color="orange" />
        <StatCard label="Connected Schools" subLabel="联网学校总数" value="482" trend="Stable" icon={Building2} color="blue" />
        <StatCard label="Medical Pressure" subLabel="医疗资源压力" value="High / 高" trend="Alert" icon={Stethoscope} color="red" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
        <div className="lg:col-span-2 bento-card space-y-8">
          <h3 className="text-2xl font-black flex items-center gap-3">
            <TrendingUp className="w-6 h-6 text-blue-600" /> 
            <div className="flex flex-col">
              <span>Regional Risk Map</span>
              <span className="text-sm opacity-50">区域风险分布图</span>
            </div>
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {MOCK_DISTRICT_DATA.map(district => (
              <div key={district.name} className="p-8 bg-slate-50 rounded-[32px] border border-slate-100 space-y-6 group hover:bg-white hover:shadow-xl transition-all">
                <div className="flex items-center justify-between">
                  <div className="font-black text-slate-900 text-xl">{district.name}</div>
                  <div className={cn(
                    "px-3 py-1 rounded-lg text-[10px] font-black uppercase tracking-widest",
                    district.riskScore > 70 ? "bg-red-100 text-red-600" : "bg-orange-100 text-orange-600"
                  )}>{district.trend}</div>
                </div>
                <div className="flex items-end justify-between">
                  <div>
                    <div className="text-[10px] text-slate-400 font-black uppercase tracking-widest mb-1">Risk / 风险评分</div>
                    <div className="text-4xl font-black text-slate-900 tracking-tighter">{district.riskScore}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-[10px] text-slate-400 font-black uppercase tracking-widest mb-1">Subjects / 监测对象</div>
                    <div className="text-lg font-black text-slate-700">{district.activeCases}</div>
                  </div>
                </div>
                <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                  <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: `${district.riskScore}%` }}
                    className={cn(
                      "h-full transition-all duration-1000",
                      district.riskScore > 70 ? "bg-red-500" : "bg-orange-500"
                    )} 
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-10">
          <div className="bento-card bg-red-50 border-red-100 space-y-6">
            <h3 className="text-xl font-black flex items-center gap-3 text-red-600">
              <Bell className="w-6 h-6" /> 
              <div className="flex flex-col">
                <span>Regional Orders</span>
                <span className="text-sm opacity-50">区域预警指令</span>
              </div>
            </h3>
            <div className="space-y-4">
              <div className="p-6 bg-white rounded-3xl border border-red-100 shadow-sm space-y-4">
                <div className="text-sm font-black text-red-600">Global Disinfection / 发布全市消杀指令</div>
                <p className="text-xs text-slate-500 leading-relaxed font-medium">High risk detected. Deep disinfection recommended.<br/><span className="text-[10px] opacity-60">检测到和平区、沈北新区风险激增，建议全市学校统一进行深度消杀。</span></p>
                <button className="w-full py-3 bg-red-600 text-white rounded-xl text-xs font-black hover:bg-red-700 transition-all shadow-lg shadow-red-100">
                  Publish Now / 立即发布
                </button>
              </div>
              <div className="p-6 bg-white rounded-3xl border border-slate-100 shadow-sm space-y-4">
                <div className="text-sm font-black text-slate-900">Resource Allocation / 医疗资源调配</div>
                <p className="text-xs text-slate-500 leading-relaxed font-medium">Remote guidance for high-risk schools.<br/><span className="text-[10px] opacity-60">协调沈阳市儿童医院专家组对高风险区域学校进行远程指导。</span></p>
                <button className="w-full py-3 bg-slate-900 text-white rounded-xl text-xs font-black hover:bg-slate-800 transition-all shadow-lg shadow-slate-200">
                  Start / 启动调配
                </button>
              </div>
            </div>
          </div>

          <div className="bento-card bg-slate-900 text-white space-y-6">
            <h3 className="text-xl font-black flex items-center gap-3">
              <ShieldCheck className="w-6 h-6 text-blue-400" /> 
              <div className="flex flex-col">
                <span>Emergency Status</span>
                <span className="text-sm opacity-50">应急响应状态</span>
              </div>
            </h3>
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-blue-600/20 rounded-2xl flex items-center justify-center">
                <div className="w-3 h-3 bg-blue-500 rounded-full animate-ping" />
              </div>
              <div>
                <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Current Level</div>
                <div className="text-xl font-black">Level 3 / 三级响应</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, subLabel, value, trend, icon: Icon, color }: any) {
  const colors: Record<string, string> = {
    blue: "bg-blue-600 shadow-blue-100 text-blue-600",
    red: "bg-red-600 shadow-red-100 text-red-600",
    orange: "bg-orange-500 shadow-orange-100 text-orange-500",
    green: "bg-emerald-600 shadow-emerald-100 text-emerald-600",
  };

  return (
    <div className="bento-card group overflow-hidden relative">
      <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:scale-110 transition-transform">
        <Icon className="w-24 h-24" />
      </div>
      <div className="relative z-10 space-y-4">
        <div className="flex items-center justify-between">
          <div className={cn("w-12 h-12 rounded-2xl flex items-center justify-center text-white shadow-xl", colors[color].split(' ')[0], colors[color].split(' ')[1])}>
            <Icon className="w-6 h-6" />
          </div>
          <div className={cn(
            "px-3 py-1 rounded-lg text-[10px] font-black uppercase tracking-widest",
            trend.startsWith('+') ? "bg-red-50 text-red-600" : "bg-emerald-50 text-emerald-600"
          )}>
            {trend}
          </div>
        </div>
        <div>
          <div className="text-3xl font-black text-slate-900 tracking-tight">{value}</div>
          <div className="flex flex-col mt-1">
            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest leading-tight">{label}</div>
            {subLabel && <div className="text-[8px] font-bold text-slate-300 uppercase tracking-widest leading-tight">{subLabel}</div>}
          </div>
        </div>
      </div>
    </div>
  );
}

function Monitor() {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [events, setEvents] = useState<CoughEvent[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<any>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  const startMonitoring = () => {
    setIsMonitoring(true);
    // 模拟音频可视化
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        let offset = 0;
        const animate = () => {
          if (!isMonitoring) return;
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.beginPath();
          ctx.strokeStyle = '#3b82f6';
          ctx.lineWidth = 2;
          for (let i = 0; i < canvas.width; i++) {
            const y = canvas.height / 2 + Math.sin(i * 0.05 + offset) * 20 * Math.random();
            if (i === 0) ctx.moveTo(i, y);
            else ctx.lineTo(i, y);
          }
          ctx.stroke();
          offset += 0.2;
          requestAnimationFrame(animate);
        };
        animate();
      }
    }

    // 模拟检测到事件
    setTimeout(() => {
      const newEvent: CoughEvent = {
        id: Date.now().toString(),
        studentName: 'Zhang Xiaoming / 张小明',
        timestamp: new Date(),
        type: 'dry',
        intensity: 85,
        duration: 1.2,
        isMultiPerson: false,
        confidence: 94.2,
        riskLevel: 'Medium',
        assessment: 'High-frequency dry cough detected. Voiceprint matches student history.\n检测到高频干咳，声纹特征与该学生历史记录匹配。',
        advice: 'Drink warm water and monitor for fever.\n建议多饮温水，观察是否有发热症状。',
        envData: { temperature: 22, humidity: 45, co2: 850 }
      };
      setEvents(prev => [newEvent, ...prev]);
    }, 3000);
  };

  return (
    <div className="space-y-12 max-w-[1600px] mx-auto">
      <header className="flex flex-col lg:flex-row lg:items-end justify-between gap-6">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-blue-600 font-black text-xs uppercase tracking-widest">
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" />
            Real-time Audio Analysis
          </div>
          <h2 className="text-5xl font-black text-slate-900 tracking-tight">
            Voiceprint Cough Monitor<br />
            <span className="text-slate-400 font-serif italic text-4xl">声纹咳嗽监测</span>
          </h2>
          <p className="text-slate-500 font-medium">Edge AI real-time classroom monitoring.<br/><span className="text-xs opacity-60">利用边缘计算与声纹识别技术实时监测班级咳嗽事件</span></p>
        </div>
        <div className="flex gap-4">
          {!isMonitoring ? (
            <button 
              onClick={startMonitoring}
              className="px-8 py-4 bg-blue-600 text-white rounded-2xl font-black text-sm hover:bg-blue-700 transition-all shadow-xl shadow-blue-100 flex items-center gap-3"
            >
              <Mic className="w-5 h-5" /> 开启实时监测 / Start Monitor
            </button>
          ) : (
            <button 
              onClick={() => setIsMonitoring(false)}
              className="px-8 py-4 bg-red-600 text-white rounded-2xl font-black text-sm hover:bg-red-700 transition-all shadow-xl shadow-red-100 flex items-center gap-3"
            >
              <Square className="w-5 h-5" /> 停止监测 / Stop Monitor
            </button>
          )}
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
        <div className="lg:col-span-2 space-y-10">
          <div className="bento-card overflow-hidden bg-slate-900 h-[400px] relative flex flex-col items-center justify-center border-none shadow-2xl shadow-slate-900/20">
            <div className="absolute inset-0 bg-gradient-to-b from-blue-600/10 to-transparent pointer-events-none" />
            <div className="absolute top-8 left-8 flex items-center gap-3 z-10">
              <div className={cn("w-3 h-3 rounded-full", isMonitoring ? "bg-red-500 animate-pulse shadow-[0_0_12px_rgba(239,68,68,0.5)]" : "bg-slate-700")} />
              <span className="text-xs font-black text-slate-400 uppercase tracking-widest">
                {isMonitoring ? (isDemoMode ? 'Demo Mode Active' : 'Live Audio Stream') : 'System Standby'}
              </span>
            </div>
            
            <canvas ref={canvasRef} width={800} height={200} className="w-full h-48 opacity-60 relative z-10" />
            
            {!isMonitoring && (
              <div className="text-center space-y-6 relative z-10">
                <div className="w-24 h-24 bg-slate-800/50 backdrop-blur-md rounded-[40px] flex items-center justify-center mx-auto border border-white/5">
                  <MicOff className="w-10 h-10 text-slate-600" />
                </div>
                <div className="space-y-2">
                  <p className="text-slate-400 font-bold">麦克风未开启 / Mic Standby</p>
                  <p className="text-slate-600 text-xs font-medium">点击上方按钮开启麦克风权限以开始监测</p>
                </div>
              </div>
            )}

            {isAnalyzing && (
              <div className="absolute inset-0 bg-slate-900/80 backdrop-blur-sm flex items-center justify-center z-20">
                <div className="text-center space-y-6">
                  <div className="relative">
                    <div className="w-24 h-24 border-4 border-blue-600/20 border-t-blue-600 rounded-full animate-spin mx-auto" />
                    <BrainCircuit className="w-10 h-10 text-blue-500 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                  </div>
                  <div className="space-y-2">
                    <div className="text-xl font-black text-white">Gemma AI 本地深度分析中...</div>
                    <div className="text-xs text-slate-400 font-bold uppercase tracking-widest">Local Analysis: Cough Pattern & Voiceprint</div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="space-y-6">
            <h3 className="text-2xl font-black flex items-center gap-3">
              <HistoryIcon className="w-6 h-6 text-blue-600" /> 
              <div className="flex flex-col">
                <span>Real-time Events</span>
                <span className="text-sm opacity-50">实时事件流</span>
              </div>
            </h3>
            <div className="space-y-4">
              <AnimatePresence initial={false}>
                {events.length === 0 ? (
                  <div className="p-12 text-center border-2 border-dashed border-slate-100 rounded-[40px]">
                    <p className="text-slate-400 font-medium italic">暂无检测到的咳嗽事件 / No events detected yet</p>
                  </div>
                ) : (
                  events.map((event) => (
                    <motion.div 
                      key={event.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="bento-card group hover:border-blue-200 transition-all"
                    >
                      <div className="flex flex-col md:flex-row gap-8">
                        <div className="flex-1 space-y-6">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-4">
                              <div className="w-12 h-12 bg-blue-600 rounded-2xl flex items-center justify-center text-white shadow-lg shadow-blue-100">
                                <UserIcon className="w-6 h-6" />
                              </div>
                              <div>
                                <div className="font-black text-slate-900 text-lg">{event.studentName}</div>
                                <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">声纹匹配度: {event.confidence.toFixed(1)}%</div>
                              </div>
                            </div>
                            <div className={cn(
                              "px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest",
                              event.riskLevel === 'High' ? "bg-red-600 text-white shadow-lg shadow-red-100" : 
                              event.riskLevel === 'Medium' ? "bg-orange-500 text-white shadow-lg shadow-orange-100" : 
                              "bg-emerald-500 text-white shadow-lg shadow-emerald-100"
                            )}>
                              {event.riskLevel} Risk
                            </div>
                          </div>

                          <div className="grid grid-cols-3 gap-4">
                            <div className="p-4 bg-slate-50 rounded-2xl space-y-1">
                              <div className="text-[10px] font-bold text-slate-400 uppercase">Cough Type</div>
                              <div className="text-sm font-black text-slate-900 capitalize">{event.type}</div>
                            </div>
                            <div className="p-4 bg-slate-50 rounded-2xl space-y-1">
                              <div className="text-[10px] font-bold text-slate-400 uppercase">Intensity</div>
                              <div className="text-sm font-black text-slate-900">{event.intensity.toFixed(0)}%</div>
                            </div>
                            <div className="p-4 bg-slate-50 rounded-2xl space-y-1">
                              <div className="text-[10px] font-bold text-slate-400 uppercase">Multi-person</div>
                              <div className="text-sm font-black text-slate-900">{event.isMultiPerson ? 'Yes' : 'No'}</div>
                            </div>
                          </div>

                          <div className="p-6 bg-blue-50 rounded-3xl border border-blue-100 space-y-3">
                            <div className="flex items-center gap-2 text-blue-600 font-black text-xs uppercase tracking-widest">
                              <BrainCircuit className="w-4 h-4" /> AI Assessment
                            </div>
                            <p className="text-sm text-slate-700 leading-relaxed font-medium whitespace-pre-wrap">{event.assessment}</p>
                            <div className="pt-3 border-t border-blue-100 text-xs text-blue-500 font-bold italic whitespace-pre-wrap">
                              Advice: {event.advice}
                            </div>
                          </div>
                        </div>
                        
                        <div className="md:w-48 space-y-4">
                          <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Environment</div>
                          <div className="space-y-3">
                            <div className="flex justify-between items-center text-xs">
                              <span className="text-slate-500 font-medium">Temp</span>
                              <span className="font-black text-slate-900">{event.envData.temperature}°C</span>
                            </div>
                            <div className="flex justify-between items-center text-xs">
                              <span className="text-slate-500 font-medium">Humidity</span>
                              <span className="font-black text-slate-900">{event.envData.humidity}%</span>
                            </div>
                            <div className="flex justify-between items-center text-xs">
                              <span className="text-slate-500 font-medium">CO2</span>
                              <span className="font-black text-slate-900">{event.envData.co2}ppm</span>
                            </div>
                          </div>
                          <div className="pt-4 text-[10px] text-slate-400 font-bold text-right">
                            {event.timestamp.toLocaleTimeString()}
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  ))
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>

        <div className="space-y-10">
          <div className="bento-card space-y-6">
            <h3 className="text-xl font-black flex items-center gap-3">
              <ShieldCheck className="w-6 h-6 text-emerald-500" /> 
              <div className="flex flex-col">
                <span>System Check</span>
                <span className="text-sm opacity-50">系统自检</span>
              </div>
            </h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-slate-50 rounded-2xl">
                <span className="text-sm font-medium text-slate-600">Audio Engine</span>
                <span className="text-xs font-black text-emerald-600 uppercase tracking-widest">Optimal</span>
              </div>
              <div className="flex items-center justify-between p-4 bg-slate-50 rounded-2xl">
                <span className="text-sm font-medium text-slate-600">YAMNet Model</span>
                <span className="text-xs font-black text-emerald-600 uppercase tracking-widest">Loaded</span>
              </div>
              <div className="flex items-center justify-between p-4 bg-slate-50 rounded-2xl">
                <span className="text-sm font-medium text-slate-600">Gemma Edge</span>
                <span className="text-xs font-black text-emerald-600 uppercase tracking-widest">Running</span>
              </div>
            </div>
          </div>

          <div className="bento-card bg-slate-900 text-white space-y-6">
            <h3 className="text-xl font-black flex items-center gap-3">
              <BrainCircuit className="w-6 h-6 text-blue-400" /> 
              <div className="flex flex-col">
                <span>Edge AI Advantages</span>
                <span className="text-sm opacity-50">边缘计算优势</span>
              </div>
            </h3>
            <p className="text-sm text-slate-400 leading-relaxed">
              Preliminary cough recognition and voiceprint extraction are performed locally. The system leverages locally deployed Gemma models for deep analysis, ensuring that sensitive information never leaves the local environment, providing maximum privacy and ultra-low latency.<br/>
              <span className="text-xs opacity-60">系统在本地完成初步咳嗽识别与声纹提取，并利用本地化部署的 Gemma 模型进行深度分析。所有敏感信息均不出本地，确保了极致的隐私保护与超低延迟。</span>
            </p>
            <div className="pt-4 border-t border-slate-800">
              <div className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-2">Local Processing Power</div>
              <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full bg-blue-600 w-3/4" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Module-level cache so a generated report survives tab switches (component unmount).
// Key: user.role. Cleared on full page reload.
const __agentReportCache: Record<string, { report: AgentReport; generatedAt: string | null }> = {};

function AgenticDashboard({ user, classrooms }: { key?: string, user: UserProfile, classrooms: Classroom[] }) {
  const cachedEntry = __agentReportCache[user.role];
  const [report, setReport] = useState<AgentReport | null>(cachedEntry?.report ?? null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [executedActions, setExecutedActions] = useState<Set<number>>(new Set());
  const [generatedAt, setGeneratedAt] = useState<string | null>(cachedEntry?.generatedAt ?? null);
  const envData = { temp: -2, humidity: 35, aqi: 42, co2: 1150 };

  // Load from cache (instant) → fallback to full generation if cache miss
  const loadReport = async (forceRefresh = false) => {
    // In-memory hit: skip network entirely
    if (!forceRefresh && __agentReportCache[user.role]) {
      const c = __agentReportCache[user.role];
      setReport(c.report);
      setGeneratedAt(c.generatedAt);
      return;
    }
    setIsGenerating(true);
    try {
      if (!forceRefresh) {
        // Try backend cache — instant response
        const cached = await fetch(`${FLUGUARD_API}/api/report/cached/${user.role}`);
        if (cached.status === 200) {
          const data = await cached.json();
          const r: AgentReport = {
            id: 'r1',
            timestamp: new Date(),
            risk_level: data.risk_level ?? (data.riskScore > 70 ? 'High' : data.riskScore > 40 ? 'Medium' : 'Low'),
            reason: data.reason ?? data.summary ?? '',
            prediction: data.prediction,
            actions: data.actions,
            data_used: data.data_used ?? [],
            riskScore: data.riskScore,
          };
          setReport(r);
          setGeneratedAt(data.generated_at ?? null);
          __agentReportCache[user.role] = { report: r, generatedAt: data.generated_at ?? null };
          return;
        }
        // 202 = still generating in background, fall through to full call
      }
      // Full generation (manual refresh or cache miss)
      const reportResult = await callFluGuardReport({
        role: user.role,
        system_prompt: '',
        env_data: envData,
        classrooms,
      });
      const r: AgentReport = {
        id: 'r1',
        timestamp: new Date(),
        risk_level: reportResult.risk_level ?? (reportResult.riskScore > 70 ? 'High' : reportResult.riskScore > 40 ? 'Medium' : 'Low'),
        reason: reportResult.reason ?? '',
        prediction: reportResult.prediction,
        actions: reportResult.actions,
        data_used: reportResult.data_used ?? [],
        riskScore: reportResult.riskScore,
      };
      const ts = new Date().toISOString();
      setReport(r);
      setGeneratedAt(ts);
      __agentReportCache[user.role] = { report: r, generatedAt: ts };
    } catch (error) {
      console.error('Agent report failed:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const executeAction = (index: number) => {
    setExecutedActions(prev => new Set(prev).add(index));
  };

  // Load cached report on mount / when role changes.
  // If module-level cache already has a report for this role, swap it in instantly
  // (no network, no loading flicker). Otherwise fetch.
  useEffect(() => {
    const cached = __agentReportCache[user.role];
    if (cached) {
      setReport(cached.report);
      setGeneratedAt(cached.generatedAt);
      setIsGenerating(false);
      return;
    }
    // No cache for this role → clear stale report from previous role and fetch
    setReport(null);
    setGeneratedAt(null);
    loadReport();
  }, [user.role]);

  const forecastData = [
    { time: '00:00', actual: 120, forecast: 120 },
    { time: '04:00', actual: 80, forecast: 80 },
    { time: '08:00', actual: 450, forecast: 450 },
    { time: '12:00', actual: 320, forecast: 320 },
    { time: '16:00', actual: null, forecast: 480 },
    { time: '20:00', actual: null, forecast: 620 },
    { time: '24:00', actual: null, forecast: 550 },
  ];

  return (
    <div className="space-y-8 max-w-[1600px] mx-auto">

      {/* ── 调度说明横幅 / Schedule Notice ── */}
      <div className="flex items-start gap-4 px-6 py-4 bg-blue-50 border border-blue-100 rounded-[24px]">
        <div className="w-8 h-8 bg-blue-600 rounded-xl flex items-center justify-center shrink-0 mt-0.5">
          <Clock className="w-4 h-4 text-white" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-xs font-black text-blue-700 uppercase tracking-widest mb-1">
            Scheduled AI Analysis / 定时 AI 分析
          </div>
          <p className="text-sm text-blue-800 font-medium leading-relaxed">
            报告基于历史监测数据，由 Gemma 4 每日 <strong>08:00 · 12:00 · 16:00</strong> 自动生成并缓存，点击菜单即时加载。如需立即刷新可点击右侧按钮重新生成。
          </p>
          <p className="text-xs text-blue-600 mt-0.5">
            Reports are auto-generated by Gemma 4 at <strong>08:00 · 12:00 · 16:00</strong> daily from historical monitoring data and served instantly from cache. Click "Refresh" to regenerate on demand.
          </p>
        </div>
        {generatedAt && (
          <div className="shrink-0 text-right">
            <div className="text-[10px] text-blue-500 font-bold uppercase tracking-widest">Last generated</div>
            <div className="text-xs font-black text-blue-700">
              {new Date(generatedAt).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}
            </div>
          </div>
        )}
      </div>

      <header className="flex flex-col lg:flex-row lg:items-end justify-between gap-6">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-blue-600 font-black text-xs uppercase tracking-widest">
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" />
            Gemma 4 · RAG · Function Calling
          </div>
          <h2 className="text-5xl font-black text-slate-900 tracking-tight">
            AI Decision Report<br />
            <span className="text-slate-400 font-serif italic text-4xl">Gemma 综合决策报告</span>
          </h2>
          <p className="text-slate-500 font-medium text-sm">
            {user.role === 'teacher' && '班级健康管理 · 学生个体趋势 · 即时可执行建议'}
            {user.role === 'principal' && '全校风险总览 · 空间布局分析 · 校级应急决策'}
            {user.role === 'bureau' && '区域流感监测 · 政策导向分析 · 资源调配建议'}
            {user.role === 'parent' && '孩子健康关怀 · 家庭防护建议 · 就医引导'}
            {!['teacher','principal','bureau','parent'].includes(user.role) && 'Multi-modal data fusion reasoning / 多模态数据融合推理'}
          </p>
        </div>
        <button
          onClick={() => loadReport(true)}
          disabled={isGenerating}
          className="px-8 py-4 bg-slate-900 text-white rounded-2xl font-black text-sm hover:bg-slate-800 transition-all shadow-xl shadow-slate-200 flex items-center gap-3 disabled:opacity-50"
        >
          {isGenerating ? <Loader2 className="w-5 h-5 animate-spin" /> : <BrainCircuit className="w-5 h-5" />}
          <div className="flex flex-col items-start">
            <span>Refresh Report</span>
            <span className="text-[10px] opacity-60">立即重新生成</span>
          </div>
        </button>
      </header>

      {envData && user.role !== 'bureau' && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {[
            { label: 'Temperature / 温度', value: `${envData.temp}°C`, icon: Thermometer, color: 'text-blue-600', bg: 'bg-blue-50' },
            { label: 'Humidity / 湿度', value: `${envData.humidity}%`, icon: Droplets, color: 'text-cyan-600', bg: 'bg-cyan-50' },
            { label: 'AQI / 空气质量', value: envData.aqi, icon: Wind, color: 'text-emerald-600', bg: 'bg-emerald-50' },
            { label: 'CO2 / 二氧化碳', value: `${envData.co2}ppm`, icon: Activity, color: 'text-orange-600', bg: 'bg-orange-50' },
          ].map((item, i) => (
            <motion.div 
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              className="p-6 bg-white rounded-[32px] border border-slate-100 shadow-sm flex items-center gap-4"
            >
              <div className={cn("w-12 h-12 rounded-2xl flex items-center justify-center", item.bg, item.color)}>
                <item.icon className="w-6 h-6" />
              </div>
              <div>
                <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest">{item.label}</div>
                <div className="text-xl font-black text-slate-900">{item.value}</div>
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {user.role === 'bureau' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            { label: 'Total Schools / 区域学校总数', value: '124', icon: Building2, color: 'text-blue-600', bg: 'bg-blue-50' },
            { label: 'High Risk Schools / 高风险学校', value: '8', icon: ShieldAlert, color: 'text-red-600', bg: 'bg-red-50' },
            { label: 'Regional Risk / 区域综合风险', value: 'Medium / 中', icon: BarChart3, color: 'text-orange-600', bg: 'bg-orange-50' },
          ].map((item, i) => (
            <motion.div 
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              className="p-8 bg-white rounded-[40px] border border-slate-100 shadow-sm flex items-center gap-6"
            >
              <div className={cn("w-16 h-16 rounded-3xl flex items-center justify-center", item.bg, item.color)}>
                <item.icon className="w-8 h-8" />
              </div>
              <div>
                <div className="text-xs font-black text-slate-400 uppercase tracking-widest">{item.label}</div>
                <div className="text-3xl font-black text-slate-900">{item.value}</div>
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {isGenerating && !report ? (
        <div className="p-20 text-center space-y-8 bento-card">
          <div className="relative">
            <div className="w-32 h-32 border-4 border-blue-600/20 border-t-blue-600 rounded-full animate-spin mx-auto" />
            <BrainCircuit className="w-12 h-12 text-blue-500 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
          </div>
          <div className="space-y-2">
            <div className="text-2xl font-black text-slate-900">Gemma AI Reasoning...</div>
            <div className="text-xs text-slate-400 font-bold uppercase tracking-widest">Fusing Multi-modal Data / 多模态数据融合推理中</div>
          </div>
        </div>
      ) : report && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
          <div className="lg:col-span-2 space-y-8">
            <div className="bento-card space-y-6 relative overflow-hidden">
              <div className="absolute top-0 right-0 p-8 opacity-5 pointer-events-none">
                <BrainCircuit className="w-56 h-56" />
              </div>
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-blue-600 rounded-2xl flex items-center justify-center text-white shadow-xl shadow-blue-100">
                  <Sparkles className="w-6 h-6" />
                </div>
                <div className="flex flex-col">
                  <h3 className="text-2xl font-black text-slate-900 tracking-tight">AI Daily Report / AI 每日决策报告</h3>
                  <span className="text-xs text-slate-400 font-bold uppercase tracking-widest">
                    Gemma 4 · RAG · Function Calling · {user.role === 'teacher' ? '班主任视角' : user.role === 'principal' ? '校长视角' : user.role === 'bureau' ? '教育局视角' : '家长视角'}
                  </span>
                </div>
              </div>

              <div className="space-y-5 relative z-10">

                {/* ── Risk Level Badge ── */}
                <div className="flex items-center gap-3">
                  <div className="text-xs font-black text-slate-400 uppercase tracking-widest">Risk Level / 风险等级</div>
                  <span className={cn(
                    "px-4 py-1.5 rounded-full text-sm font-black uppercase tracking-widest",
                    report.risk_level === 'Critical' ? "bg-red-600 text-white" :
                    report.risk_level === 'High'     ? "bg-red-100 text-red-700" :
                    report.risk_level === 'Medium'   ? "bg-orange-100 text-orange-700" :
                                                       "bg-emerald-100 text-emerald-700"
                  )}>
                    {report.risk_level}
                  </span>
                </div>

                {/* ── Reason ── */}
                <div className="space-y-2">
                  <div className="text-xs font-black text-slate-400 uppercase tracking-widest flex items-center gap-2">
                    <Activity className="w-4 h-4" /> Reason / 原因分析
                  </div>
                  <p className="text-base font-medium text-slate-700 leading-relaxed whitespace-pre-wrap">
                    {report.reason}
                  </p>
                </div>

                {/* ── Data Used (traceability) ── */}
                {report.data_used && report.data_used.length > 0 && (
                  <div className="p-5 bg-slate-50 rounded-[20px] border border-slate-100 space-y-2">
                    <div className="text-xs font-black text-slate-400 uppercase tracking-widest flex items-center gap-2">
                      <Database className="w-4 h-4" /> Data Used / 数据来源
                    </div>
                    <div className="font-mono text-xs text-slate-600 space-y-1">
                      {report.data_used.map((d, i) => (
                        <div key={i} className="flex gap-2">
                          <span className="text-blue-400 shrink-0">▸</span>
                          <span className="whitespace-pre-wrap break-all">{d}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* ── 24-48h Prediction ── */}
                <div className="p-5 bg-blue-50 rounded-[20px] border border-blue-100 space-y-2">
                  <div className="text-xs font-black text-blue-600 uppercase tracking-widest flex items-center gap-2">
                    <TrendingUp className="w-4 h-4" /> 24–48h Outlook / 趋势预测
                  </div>
                  <p className="text-sm text-slate-600 leading-relaxed font-semibold whitespace-pre-wrap">{report.prediction}</p>
                </div>
              </div>
            </div>

            <div className="bento-card space-y-6">
              <h3 className="text-xl font-black flex items-center gap-3">
                <Zap className="w-5 h-5 text-orange-500" />
                <div className="flex flex-col">
                  <span>Recommended Actions / 建议执行操作</span>
                  <span className="text-xs opacity-50 font-bold">点击"立即执行"标记已下达 · Click to mark as dispatched</span>
                </div>
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {report.actions.map((action, i) => (
                  <div key={i} className="p-5 bg-slate-50 rounded-2xl border border-slate-100 space-y-3 group hover:bg-white hover:shadow-lg transition-all">
                    <p className="text-sm font-semibold text-slate-700 leading-relaxed">{action}</p>
                    <button
                      onClick={() => executeAction(i)}
                      disabled={executedActions.has(i)}
                      className={cn(
                        "w-full py-2.5 rounded-xl font-black text-xs transition-all flex items-center justify-center gap-2",
                        executedActions.has(i)
                          ? "bg-emerald-600 text-white cursor-default"
                          : "bg-slate-900 text-white hover:bg-slate-800"
                      )}
                    >
                      {executedActions.has(i) ? (
                        <><CheckCircle2 className="w-4 h-4" /> Executed / 已下达</>
                      ) : (
                        <><Play className="w-4 h-4" /> Execute Now / 立即执行</>
                      )}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-10">
            <div className="bento-card space-y-8 text-center bg-gradient-to-br from-white to-slate-50 border-none shadow-2xl shadow-slate-200/50">
              <h3 className="text-xl font-black text-slate-900">Overall Risk Score<br /><span className="text-slate-400 text-xs uppercase tracking-widest">当前综合风险评分</span></h3>
              <div className="relative inline-block">
                <svg className="w-56 h-56 transform -rotate-90">
                  <defs>
                    <linearGradient id="riskGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#10b981" />
                      <stop offset="50%" stopColor="#f59e0b" />
                      <stop offset="100%" stopColor="#ef4444" />
                    </linearGradient>
                  </defs>
                  <circle cx="112" cy="112" r="90" stroke="currentColor" strokeWidth="16" fill="transparent" className="text-slate-100" />
                  <motion.circle 
                    cx="112" cy="112" r="90" stroke="url(#riskGradient)" strokeWidth="16" fill="transparent" 
                    strokeDasharray={565.5}
                    initial={{ strokeDashoffset: 565.5 }}
                    animate={{ strokeDashoffset: 565.5 - (565.5 * report.riskScore) / 100 }}
                    strokeLinecap="round"
                    className="transition-all duration-1000"
                  />
                </svg>
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-center">
                  <motion.div 
                    initial={{ scale: 0.5, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="text-6xl font-black text-slate-900 tracking-tighter"
                  >
                    {report.riskScore}
                  </motion.div>
                  <div className={cn(
                    "px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest mt-2",
                    report.riskScore > 70 ? "bg-red-100 text-red-600" : report.riskScore > 40 ? "bg-orange-100 text-orange-600" : "bg-emerald-100 text-emerald-600"
                  )}>
                    {report.riskScore > 70 ? 'High Risk' : report.riskScore > 40 ? 'Medium Risk' : 'Low Risk'}
                  </div>
                </div>
              </div>
              <p className="text-sm font-medium text-slate-500">Based on cough frequency, CO2 levels, and epidemic data.<br/><span className="text-[10px] opacity-60">基于全校咳嗽频率、环境 CO2 浓度及外部疫情数据综合计算。</span></p>
              <div className="pt-6 border-t border-slate-100 grid grid-cols-2 gap-4">
                <div className="text-left">
                  <div className="text-[10px] font-black text-slate-400 uppercase">Confidence</div>
                  <div className="text-lg font-black text-slate-900">98.2%</div>
                </div>
                <div className="text-right">
                  <div className="text-[10px] font-black text-slate-400 uppercase">Status</div>
                  <div className="text-lg font-black text-emerald-600">Stable</div>
                </div>
              </div>
            </div>

            <div className="bento-card space-y-6">
              <h3 className="text-xl font-black flex items-center gap-3">
                <Wind className="w-6 h-6 text-blue-600" /> 
                <div className="flex flex-col">
                  <span>Env Context</span>
                  <span className="text-sm opacity-50">环境上下文</span>
                </div>
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-blue-50/50 rounded-2xl border border-blue-100/50 space-y-1">
                  <div className="flex items-center gap-2 text-[10px] font-black text-blue-600 uppercase">
                    <Thermometer className="w-3 h-3" /> Temp
                  </div>
                  <div className="text-xl font-black text-slate-900">{envData?.temp}°C</div>
                </div>
                <div className="p-4 bg-emerald-50/50 rounded-2xl border border-emerald-100/50 space-y-1">
                  <div className="flex items-center gap-2 text-[10px] font-black text-emerald-600 uppercase">
                    <Droplets className="w-3 h-3" /> Humidity
                  </div>
                  <div className="text-xl font-black text-slate-900">{envData?.humidity}%</div>
                </div>
                <div className="p-4 bg-orange-50/50 rounded-2xl border border-orange-100/50 space-y-1">
                  <div className="flex items-center gap-2 text-[10px] font-black text-orange-600 uppercase">
                    <Activity className="w-3 h-3" /> AQI
                  </div>
                  <div className="text-xl font-black text-slate-900">{envData?.aqi}</div>
                </div>
                <div className="p-4 bg-purple-50/50 rounded-2xl border border-purple-100/50 space-y-1">
                  <div className="flex items-center gap-2 text-[10px] font-black text-purple-600 uppercase">
                    <Wind className="w-3 h-3" /> CO2
                  </div>
                  <div className="text-xl font-black text-slate-900">{envData?.co2} <span className="text-[10px] opacity-50">ppm</span></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ParentPortal() {
  const [messages, setMessages] = useState([
    { 
      id: 1, 
      role: 'ai', 
      text: 'Attention parents! There are more coughs in our child\'s class today. Shenyang is currently in a high flu period. Please ensure good ventilation, avoid crowded places, and consider giving your child some preventive measures. \n 爸妈注意啦，咱孩子班今天咳嗽有点多，沈阳现在是甲流高发期，多通风、少去人多地方，早上可以给孩子喝点板蓝根哦~', 
      time: '08:30' 
    },
  ]);
  const [input, setInput] = useState('');

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMsg = { id: Date.now(), role: 'user', text: input, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) };
    setMessages(prev => [...prev, userMsg]);
    setInput('');

    try {
      const systemPrompt = SYSTEM_PROMPTS.PARENT + "\n\n" + SYSTEM_PROMPTS.SAFETY;
      const result = await callFluGuardAI({
        role: 'parent',
        message: `${input}\n\nIMPORTANT: Your response MUST be bilingual. Provide English first, then Chinese below it.`,
        system_prompt: systemPrompt,
      });
      const aiMsg = {
        id: Date.now() + 1,
        role: 'ai',
        text: result.text || 'Sorry, signal is weak. Please wait. \n 哎呀，信号不好，等会儿哈。',
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, aiMsg]);
    } catch (error) {
      console.error('Parent chat failed:', error);
    }
  };

  return (
    <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="max-w-md mx-auto h-[700px] bg-white rounded-[48px] border-[8px] border-slate-900 shadow-2xl flex flex-col overflow-hidden relative">
      <div className="bg-slate-900 p-6 text-white flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-600 rounded-2xl flex items-center justify-center">
            <UserIcon className="w-6 h-6" />
          </div>
          <div>
            <div className="text-sm font-bold">FluGuard AI · Parent Portal / 家长端</div>
            <div className="text-[10px] text-emerald-400 flex items-center gap-1">
              <div className="w-1 h-1 bg-emerald-400 rounded-full animate-pulse" />
              Gemma AI Local / 本地
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-slate-50">
        {messages.map((msg) => (
          <div key={msg.id} className={cn("flex flex-col", msg.role === 'user' ? "items-end" : "items-start")}>
          <div className={cn(
            "max-w-[85%] p-4 rounded-3xl text-sm leading-relaxed shadow-sm whitespace-pre-wrap",
            msg.role === 'user' ? "bg-blue-600 text-white rounded-tr-none" : "bg-white text-slate-900 rounded-tl-none border border-slate-100"
          )}>
            {msg.text}
          </div>
            <span className="text-[10px] text-slate-400 mt-1 px-2">{msg.time}</span>
          </div>
        ))}
      </div>

      <div className="p-4 bg-white border-t border-slate-100 flex gap-2">
        <input 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Ask Gemma... / 问问 Gemma 孩子咳嗽咋办..."
          className="flex-1 bg-slate-100 border-none rounded-2xl px-4 py-3 text-sm focus:ring-2 focus:ring-blue-600 transition-all outline-none"
        />
        <button onClick={sendMessage} className="w-12 h-12 bg-blue-600 text-white rounded-2xl flex items-center justify-center shadow-lg shadow-blue-100 hover:bg-blue-700 transition-all">
          <Send className="w-5 h-5" />
        </button>
      </div>
    </motion.div>
  );
}

function Alerts({ user }: { user: UserProfile }) {
  const allAlerts = [
    { 
      id: 1, 
      type: 'critical', 
      title: 'Flu Outbreak Warning / 流感爆发预警', 
      content: 'Cough frequency in Class 1-1 exceeds threshold (150/24h). Immediate isolation and classroom disinfection recommended. \n 一年级一班咳嗽频率超过阈值 (150次/24h)，建议立即采取隔离措施并进行全班消杀。', 
      time: '10m ago / 10分钟前',
      roles: ['admin', 'teacher', 'principal', 'bureau', 'doctor']
    },
    { 
      id: 2, 
      type: 'warning', 
      title: 'Regional Epidemic Trend / 区域流行趋势', 
      content: 'Influenza index in Shenbei District rose to 68.5. Please strengthen morning and afternoon health checks. \n 沈北新区流感流行指数上升至 68.5，请各班级加强晨午检。', 
      time: '1h ago / 1小时前',
      roles: ['admin', 'teacher', 'principal', 'bureau', 'doctor']
    },
    { 
      id: 3, 
      type: 'info', 
      title: 'Prevention Reminder / 预防提醒', 
      content: 'Parents are advised to prepare spare masks for children and sync with the latest flu prevention plan. \n 建议家长为孩子准备备用口罩，并同步最新的流感预防方案。', 
      time: '3h ago / 3小时前',
      roles: ['admin', 'parent']
    },
    {
      id: 4,
      type: 'warning',
      title: 'Health Tip / 健康小贴士',
      content: 'Temperature changes are large recently. Remember to add or remove clothes in time and drink more warm water! \n 最近气温变化大，记得及时增减衣物，多喝温开水哦！',
      time: 'Just now / 刚刚',
      roles: ['student']
    },
    {
      id: 5,
      type: 'info',
      title: 'School Notice / 学校通知',
      content: 'There will be a school-wide cleaning this Friday afternoon. Please keep your classroom tidy. \n 本周五下午将进行全校大扫除，请同学们保持教室整洁。',
      time: '2h ago / 2小时前',
      roles: ['student', 'teacher']
    },
    {
      id: 6,
      type: 'critical',
      title: 'Urgent Notice / 紧急通知',
      content: 'Due to the increased flu alert level, please wear masks at all times while in school. \n 因流感预警等级提升，请同学们在校期间全程佩戴口罩。',
      time: '5h ago / 5小时前',
      roles: ['student', 'parent', 'teacher']
    }
  ];

  const alerts = allAlerts.filter(a => a.roles.includes(user.role));

  return (
    <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }} className="max-w-3xl mx-auto space-y-8">
      <header className="flex items-center justify-between">
        <h2 className="text-3xl font-black tracking-tight text-slate-900">
          Alert Center<br />
          <span className="text-sm opacity-50 font-bold">预警中心</span>
        </h2>
      </header>
      <div className="space-y-4">
        {alerts.map((alert) => (
          <div key={alert.id} className={cn("p-6 rounded-3xl border shadow-sm flex gap-6 items-start transition-all hover:shadow-md", alert.type === 'critical' ? "bg-red-50 border-red-100" : alert.type === 'warning' ? "bg-orange-50 border-orange-100" : "bg-white border-slate-100")}>
            <div className={cn("w-12 h-12 rounded-2xl flex items-center justify-center shrink-0", alert.type === 'critical' ? "bg-red-500" : alert.type === 'warning' ? "bg-orange-500" : "bg-blue-500")}>
              {alert.type === 'critical' ? <ShieldAlert className="text-white w-6 h-6" /> : alert.type === 'warning' ? <AlertTriangle className="text-white w-6 h-6" /> : <Info className="text-white w-6 h-6" />}
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="font-black text-slate-900 whitespace-pre-wrap">{alert.title}</h4>
                <span className="text-[10px] text-slate-400 font-bold whitespace-pre-wrap">{alert.time}</span>
              </div>
              <p className="text-slate-600 text-sm leading-relaxed whitespace-pre-wrap">{alert.content}</p>
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

import { Stage, Layer, Rect, Text as KonvaText, Group } from 'react-konva';

function SpatialCanvas({ classrooms, setClassrooms, totalFloors }: { classrooms: Classroom[], setClassrooms: any, totalFloors: number }) {
  // 拓扑图布局逻辑：根据楼层自动分层
  const floors = Array.from({ length: totalFloors }, (_, i) => i + 1);
  
  // 动态计算间距，确保所有楼层都能放下
  const canvasHeight = 550;
  const floorSpacing = Math.min(120, (canvasHeight - 100) / totalFloors);
  const bottomMargin = 50;

  // 初始化位置：如果教室没有 x, y，则根据楼层分配初始位置
  const [rooms, setRooms] = useState(classrooms.map((c, i) => ({
    ...c,
    x: (c as any).x || 150 + (i % 3) * 200,
    y: (c as any).y || (canvasHeight - bottomMargin) - (c.floor || 1) * floorSpacing,
    width: 160,
    height: 80
  })));

  useEffect(() => {
    // 当外部 classrooms 变化时（如新增），同步更新 rooms
    setRooms(prev => classrooms.map((c, i) => {
      const existing = prev.find(r => r.id === c.id);
      return {
        ...c,
        x: existing?.x ?? 150 + (i % 3) * 200,
        y: existing?.y ?? (canvasHeight - bottomMargin) - (c.floor || 1) * floorSpacing,
        width: 160,
        height: 80
      };
    }));
  }, [classrooms, totalFloors]); // canvasHeight/bottomMargin/floorSpacing are derived from these

  const handleDragEnd = (id: string, e: any) => {
    const newRooms = rooms.map(r => {
      if (r.id === id) {
        return { ...r, x: e.target.x(), y: e.target.y() };
      }
      return r;
    });
    setRooms(newRooms);
    // 这里可以同步回父组件，但为了演示流畅，我们只在本地维护位置
  };

  return (
    <div className="bg-slate-900 rounded-[40px] border-4 border-slate-800 p-6 h-[600px] overflow-hidden relative shadow-2xl">
      <div className="absolute top-6 left-6 z-10 space-y-2">
        <div className="bg-blue-500/20 backdrop-blur px-4 py-2 rounded-xl border border-blue-500/30 text-xs font-bold text-blue-400 flex items-center gap-2">
          <Building2 className="w-4 h-4" />
          Campus Spatial Topology (Side View) / 校园空间拓扑图 (侧视图模式)
        </div>
        <div className="bg-slate-800/50 backdrop-blur px-3 py-1.5 rounded-lg text-[10px] text-slate-400">
          Tip: Drag to move. Orange lines show vertical risk. / 提示：拖拽班级调整位置。橙色虚线表示垂直传播高风险路径。
        </div>
      </div>

      <Stage width={900} height={550}>
        <Layer>
          {/* 1. 绘制楼梯井 (Staircase Shaft) */}
          <Rect x={60} y={50} width={60} height={canvasHeight - 100} fill="#1e293b" cornerRadius={10} />
          <KonvaText x={70} y={60} text="Stairs / 楼梯/电梯" fontSize={10} fill="#64748b" align="center" width={40} />
          
          {/* 2. 绘制楼层线 (Floor Corridors) */}
          {floors.map(f => (
            <Group key={f}>
              <Rect x={120} y={(canvasHeight - bottomMargin) - f * floorSpacing + 40} width={700} height={4} fill="#334155" cornerRadius={2} />
              <KonvaText x={20} y={(canvasHeight - bottomMargin) - f * floorSpacing + 35} text={`${f}F`} fontSize={14} fontStyle="bold" fill="#475569" />
            </Group>
          ))}

          {/* 3. 绘制垂直传播风险连线 (Risk Links) */}
          {rooms.filter(r => r.nearStairs).map((room, i, arr) => {
            // 寻找同一垂直线（楼梯口）上方的班级
            const aboveRoom = arr.find(r => r.floor === (room.floor || 1) + 1);
            if (aboveRoom) {
              return (
                <Group key={`link-${room.id}-${aboveRoom.id}`}>
                  <Rect 
                    x={room.x + 80} 
                    y={aboveRoom.y + 80} 
                    width={2} 
                    height={room.y - aboveRoom.y - 80} 
                    fill="#f97316" 
                    dash={[5, 5]} 
                  />
                  <KonvaText 
                    x={room.x + 90} 
                    y={(room.y + aboveRoom.y) / 2} 
                    text="垂直传播风险 / Vertical Risk" 
                    fontSize={9} 
                    fill="#f97316" 
                    fontStyle="italic"
                  />
                </Group>
              );
            }
            return null;
          })}

          {/* 4. 绘制教室节点 (Classroom Nodes) */}
          {rooms.map((room) => (
            <Group
              key={room.id}
              x={room.x}
              y={room.y}
              draggable
              onDragEnd={(e) => handleDragEnd(room.id, e)}
            >
              <Rect
                width={room.width}
                height={room.height}
                fill={room.fluRiskLevel === 'high' ? "#450a0a" : "#0f172a"}
                stroke={room.fluRiskLevel === 'high' ? "#ef4444" : room.nearStairs ? "#f97316" : "#3b82f6"}
                strokeWidth={2}
                cornerRadius={12}
                shadowBlur={room.fluRiskLevel === 'high' ? 20 : 5}
                shadowColor={room.fluRiskLevel === 'high' ? "#ef4444" : "#000"}
                shadowOpacity={0.5}
              />
              
              {/* 装饰性侧边条 */}
              <Rect 
                width={6} 
                height={room.height} 
                fill={room.fluRiskLevel === 'high' ? "#ef4444" : room.nearStairs ? "#f97316" : "#3b82f6"} 
                cornerRadius={[12, 0, 0, 12]} 
              />

              <KonvaText
                text={room.name}
                fontSize={14}
                fontStyle="bold"
                fill="white"
                x={20}
                y={15}
              />
              
              <KonvaText
                text={`${room.floor}F | ${room.nearStairs ? '楼梯口 / Stairs' : '普通区 / Normal'}`}
                fontSize={9}
                fill="#94a3b8"
                x={20}
                y={38}
              />

              <Group x={20} y={55}>
                <Rect width={120} height={4} fill="#1e293b" cornerRadius={2} />
                <Rect 
                  width={Math.min(120, (room.coughCount24h / 200) * 120)} 
                  height={4} 
                  fill={room.fluRiskLevel === 'high' ? "#ef4444" : "#3b82f6"} 
                  cornerRadius={2} 
                />
                <KonvaText 
                  text={`24h咳嗽 / Cough: ${room.coughCount24h}`} 
                  fontSize={8} 
                  fill="#64748b" 
                  y={8} 
                />
              </Group>

              {/* 风险图标 */}
              {room.fluRiskLevel === 'high' && (
                <Group x={130} y={10}>
                  <Rect width={20} height={20} fill="#ef4444" cornerRadius={6} />
                  <KonvaText text="!" fontSize={14} fontStyle="bold" fill="white" x={7} y={3} />
                </Group>
              )}
            </Group>
          ))}
        </Layer>
      </Stage>
    </div>
  );
}

function ManagementSystem({ classrooms, setClassrooms, students, setStudents, totalFloors, setTotalFloors }: { key?: string, classrooms: Classroom[], setClassrooms: any, students: any, setStudents: any, totalFloors: number, setTotalFloors: any }) {
  const [activeSubTab, setActiveSubTab] = useState<'schools' | 'classrooms' | 'students'>('classrooms');
  const [searchTerm, setSearchTerm] = useState('');
  const [showAddModal, setShowAddModal] = useState(false);
  const [newName, setNewName] = useState('');
  const [newFloor, setNewFloor] = useState(1);
  const [isNearStairs, setIsNearStairs] = useState(false);

  const deleteStudent = (id: string) => {
    setStudents(students.filter((s: any) => s.id !== id));
  };

  const deleteClassroom = (id: string) => {
    setClassrooms(classrooms.filter((c: any) => c.id !== id));
  };

  const handleAdd = () => {
    if (!newName.trim()) return;
    
    if (activeSubTab === 'students') {
      const newStudent = {
        id: `s${Date.now()}`,
        name: newName,
        classId: 'c1',
        absenceStatus: 'present',
        coughCount: 0,
        lastCough: '-',
        voiceprintEnrolled: false
      };
      setStudents([...students, newStudent]);
    } else if (activeSubTab === 'classrooms') {
      const newClass = {
        id: `c${Date.now()}`,
        name: newName,
        grade: '1',
        studentCount: 0,
        fluRiskLevel: 'low' as const,
        coughCount24h: 0,
        floor: newFloor,
        nearStairs: isNearStairs
      };
      setClassrooms([...classrooms, newClass]);
    }
    
    setNewName('');
    setNewFloor(1);
    setIsNearStairs(false);
    setShowAddModal(false);
  };

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-6xl mx-auto space-y-8 relative">
      <header className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-black tracking-tight text-slate-900">
            System Management<br />
            <span className="text-sm opacity-50 font-bold">系统管理</span>
          </h2>
          <p className="text-slate-500 mt-1">Manage layout, classes, and students<br/><span className="text-xs opacity-60">管理学校空间布局、班级及学生基础数据</span></p>
        </div>
        <button 
          onClick={() => setShowAddModal(true)}
          className="px-6 py-3 bg-blue-600 text-white rounded-2xl font-bold flex items-center gap-2 hover:bg-blue-700 transition-all"
        >
          <Plus className="w-5 h-5" />
          Add / 新增 {activeSubTab === 'schools' ? 'District / 区域' : activeSubTab === 'classrooms' ? 'Class / 班级' : 'Student / 学生'}
        </button>
      </header>

      {/* Add Modal */}
      <AnimatePresence>
        {showAddModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4">
            <motion.div 
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="bg-white rounded-[32px] p-8 max-w-md w-full shadow-2xl space-y-6"
            >
              <h3 className="text-xl font-bold">Add / 新增 {activeSubTab === 'schools' ? 'District / 区域' : activeSubTab === 'classrooms' ? 'Class / 班级' : 'Student / 学生'}</h3>
              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-xs font-bold text-slate-400 uppercase">Name / 名称 / 姓名</label>
                  <input 
                    autoFocus
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    placeholder="Input... / 请输入..."
                    className="w-full px-4 py-3 bg-slate-50 border-none rounded-xl focus:ring-2 focus:ring-blue-600 outline-none"
                  />
                </div>

                {activeSubTab === 'classrooms' && (
                  <>
                    <div className="space-y-2">
                      <label className="text-xs font-bold text-slate-400 uppercase">楼层 / Floor</label>
                      <select 
                        value={newFloor}
                        onChange={(e) => setNewFloor(parseInt(e.target.value))}
                        className="w-full px-4 py-3 bg-slate-50 border-none rounded-xl focus:ring-2 focus:ring-blue-600 outline-none"
                      >
                        {[...Array(totalFloors)].map((_, i) => <option key={i+1} value={i+1}>{i+1}F</option>)}
                      </select>
                    </div>
                    <div className="flex items-center gap-3 p-4 bg-slate-50 rounded-xl">
                      <input 
                        type="checkbox" 
                        id="nearStairs"
                        checked={isNearStairs}
                        onChange={(e) => setIsNearStairs(e.target.checked)}
                        className="w-5 h-5 rounded border-slate-300 text-blue-600 focus:ring-blue-600"
                      />
                      <label htmlFor="nearStairs" className="text-sm font-bold text-slate-600 cursor-pointer">Near Stairs (High Risk) / 临近楼梯/电梯口</label>
                    </div>
                  </>
                )}
              </div>
              <div className="flex gap-3">
                <button onClick={() => setShowAddModal(false)} className="flex-1 py-3 bg-slate-100 text-slate-600 rounded-xl font-bold hover:bg-slate-200 transition-all">Cancel / 取消</button>
                <button onClick={handleAdd} className="flex-1 py-3 bg-blue-600 text-white rounded-xl font-bold hover:bg-blue-700 transition-all">Add / 确认添加</button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      <div className="bg-white rounded-[40px] border border-slate-100 shadow-sm overflow-hidden">
        <div className="flex border-b border-slate-100">
          {[
            { id: 'schools', label: 'Layout / 空间布局', icon: Building2 },
            { id: 'classrooms', label: 'Classes / 班级管理', icon: Users },
            { id: 'students', label: 'Students / 学生管理', icon: UserIcon },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveSubTab(tab.id as any)}
              className={cn(
                "flex-1 py-6 flex items-center justify-center gap-3 font-bold transition-all border-b-2",
                activeSubTab === tab.id ? "border-blue-600 text-blue-600 bg-blue-50/30" : "border-transparent text-slate-400 hover:text-slate-600 hover:bg-slate-50"
              )}
            >
              <tab.icon className="w-5 h-5" />
              {tab.label}
            </button>
          ))}
        </div>

        <div className="p-8 space-y-6">
          {activeSubTab === 'schools' && (
            <div className="space-y-6">
              <div className="bg-blue-50 p-6 rounded-3xl border border-blue-100 flex items-center justify-between">
                <div className="space-y-1">
                  <div className="flex items-center gap-2 text-blue-700 font-bold">
                    <Building2 className="w-5 h-5" /> Building Settings / 教学楼基础设置
                  </div>
                  <p className="text-sm text-blue-600">Set floors to generate topology. / 设置学校总楼层数，系统将自动生成对应的空间拓扑结构。</p>
                </div>
                <div className="flex items-center gap-3 bg-white p-3 rounded-2xl border border-blue-200 shadow-sm">
                  <label className="text-xs font-black text-slate-400 uppercase">Floors / 总楼层数</label>
                  <div className="flex items-center gap-2">
                    <button 
                      onClick={() => setTotalFloors(Math.max(1, totalFloors - 1))}
                      className="w-8 h-8 flex items-center justify-center bg-slate-100 rounded-lg hover:bg-slate-200"
                    >-</button>
                    <span className="w-8 text-center font-bold text-blue-600">{totalFloors}</span>
                    <button 
                      onClick={() => setTotalFloors(Math.min(10, totalFloors + 1))}
                      className="w-8 h-8 flex items-center justify-center bg-slate-100 rounded-lg hover:bg-slate-200"
                    >+</button>
                  </div>
                </div>
              </div>
              <SpatialCanvas classrooms={classrooms} setClassrooms={setClassrooms} totalFloors={totalFloors} />
            </div>
          )}

          <div className="relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 w-5 h-5" />
            <input
              type="text"
              placeholder={`Search / 搜索 ${activeSubTab === 'schools' ? 'Floor / 楼层或区域' : activeSubTab === 'classrooms' ? 'Class / 班级名称' : 'Student / 学生姓名'}...`}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-12 pr-4 py-4 bg-slate-50 border-none rounded-2xl focus:ring-2 focus:ring-blue-600 outline-none transition-all"
            />
          </div>

          <div className="overflow-hidden rounded-2xl border border-slate-100">
            <table className="w-full text-left">
              <thead className="bg-slate-50 text-slate-400 text-[10px] font-black uppercase tracking-widest">
                <tr>
                  {activeSubTab === 'students' && (
                    <>
                      <th className="px-6 py-4">Name / 姓名</th>
                      <th className="px-6 py-4">Class / 所属班级</th>
                      <th className="px-6 py-4">Status / 状态</th>
                      <th className="px-6 py-4">Voiceprint / 声纹状态</th>
                      <th className="px-6 py-4">Action / 操作</th>
                    </>
                  )}
                  {activeSubTab === 'classrooms' && (
                    <>
                      <th className="px-6 py-4">Class / 班级名称</th>
                      <th className="px-6 py-4">Floor / 楼层</th>
                      <th className="px-6 py-4">Stairs / 楼梯口</th>
                      <th className="px-6 py-4">Count / 人数</th>
                      <th className="px-6 py-4">Risk / 风险</th>
                      <th className="px-6 py-4">Action / 操作</th>
                    </>
                  )}
                  {activeSubTab === 'schools' && (
                    <>
                      <th className="px-6 py-4">Floor / 楼层</th>
                      <th className="px-6 py-4">Classes / 包含班级</th>
                      <th className="px-6 py-4">Vertical Risk / 垂直风险</th>
                      <th className="px-6 py-4">Status / 状态</th>
                      <th className="px-6 py-4">Action / 操作</th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-50">
                {activeSubTab === 'students' && students.map((student: any) => (
                  <tr key={student.id} className="hover:bg-slate-50 transition-colors">
                    <td className="px-6 py-4 font-bold text-slate-900">{student.name}</td>
                    <td className="px-6 py-4 text-sm text-slate-500">Grade 1 Class 1 / 一年级一班</td>
                    <td className="px-6 py-4">
                      <span className={cn(
                        "px-2 py-1 rounded-lg text-[10px] font-bold",
                        student.absenceStatus === 'present' ? "bg-emerald-50 text-emerald-600" : "bg-red-50 text-red-600"
                      )}>
                        {student.absenceStatus === 'present' ? 'Present / 在校' : 'Absent / 缺勤'}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <div className={cn("w-2 h-2 rounded-full", student.voiceprintEnrolled ? "bg-emerald-500" : "bg-slate-300")} />
                        <span className="text-xs text-slate-500">{student.voiceprintEnrolled ? 'Enrolled / 已录入' : 'None / 未录入'}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex gap-2">
                        <button className="p-2 text-slate-400 hover:text-blue-600 transition-colors"><ChevronRight className="w-4 h-4" /></button>
                        <button onClick={() => deleteStudent(student.id)} className="p-2 text-slate-400 hover:text-red-600 transition-colors"><Trash2 className="w-4 h-4" /></button>
                      </div>
                    </td>
                  </tr>
                ))}
                {activeSubTab === 'classrooms' && classrooms.map(cls => (
                  <tr key={cls.id} className="hover:bg-slate-50 transition-colors">
                    <td className="px-6 py-4 font-bold text-slate-900">{cls.name}</td>
                    <td className="px-6 py-4 text-sm text-slate-500">
                      <select 
                        value={cls.floor || 1} 
                        onChange={(e) => {
                          const val = parseInt(e.target.value);
                          setClassrooms(classrooms.map(c => c.id === cls.id ? { ...c, floor: val } : c));
                        }}
                        className="bg-transparent border-none p-0 focus:ring-0 cursor-pointer hover:text-blue-600"
                      >
                        {[...Array(totalFloors)].map((_, i) => <option key={i+1} value={i+1}>{i+1}F</option>)}
                      </select>
                    </td>
                    <td className="px-6 py-4">
                      <button 
                        onClick={() => setClassrooms(classrooms.map(c => c.id === cls.id ? { ...c, nearStairs: !c.nearStairs } : c))}
                        className={cn(
                          "px-2 py-1 rounded-lg text-[10px] font-bold transition-all",
                          cls.nearStairs ? "bg-orange-50 text-orange-600 border border-orange-100" : "bg-slate-50 text-slate-400 border border-slate-100"
                        )}
                      >
                        {cls.nearStairs ? 'Near Stairs / 临近楼梯' : 'Normal / 普通位置'}
                      </button>
                    </td>
                    <td className="px-6 py-4 text-sm text-slate-500">{cls.studentCount} Students / 人</td>
                    <td className="px-6 py-4">
                      <span className={cn(
                        "px-2 py-1 rounded-lg text-[10px] font-bold",
                        cls.fluRiskLevel === 'high' ? "bg-red-50 text-red-600" : cls.fluRiskLevel === 'medium' ? "bg-orange-50 text-orange-600" : "bg-emerald-50 text-emerald-600"
                      )}>
                        {cls.fluRiskLevel.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex gap-2">
                        <button className="p-2 text-slate-400 hover:text-blue-600 transition-colors"><ChevronRight className="w-4 h-4" /></button>
                        <button onClick={() => deleteClassroom(cls.id)} className="p-2 text-slate-400 hover:text-red-600 transition-colors"><Trash2 className="w-4 h-4" /></button>
                      </div>
                    </td>
                  </tr>
                ))}
                {activeSubTab === 'schools' && (
                  <tr className="hover:bg-slate-50 transition-colors">
                    <td className="px-6 py-4 font-bold text-slate-900">1F (1st Floor)</td>
                    <td className="px-6 py-4 text-sm text-slate-500">Class 1, 2, 3 / 一班, 二班, 三班</td>
                    <td className="px-6 py-4 text-sm text-slate-500">High / 高 (Shared Stairs / 楼梯间共用)</td>
                    <td className="px-6 py-4">
                      <span className="px-2 py-1 rounded-lg text-[10px] font-bold bg-emerald-50 text-emerald-600">Normal / 正常</span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex gap-2">
                        <button className="p-2 text-slate-400 hover:text-blue-600 transition-colors"><ChevronRight className="w-4 h-4" /></button>
                        <button className="p-2 text-slate-400 hover:text-red-600 transition-colors"><Trash2 className="w-4 h-4" /></button>
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function VoiceprintEnrollment({ students, setStudents }: { students: any[], setStudents: React.Dispatch<React.SetStateAction<any[]>> }) {
  // ── 选中学生（默认选第一个）──────────────────────────────────────────────
  const [selectedStudent, setSelectedStudent] = useState<string>(() => students[0]?.id ?? '');

  // ── 录入状态 ─────────────────────────────────────────────────────────────
  const [enrollState, setEnrollState] = useState<'idle' | 'recording' | 'uploading' | 'done' | 'error'>('idle');
  const [enrollSecs, setEnrollSecs] = useState(0);
  const [enrollMsg, setEnrollMsg] = useState('');
  const [enrollRound, setEnrollRound] = useState(0); // 已完成的轮次 0-3
  // 已在后端录入的名字（权威来源）
  const [enrolledNames, setEnrolledNames] = useState<string[]>([]);

  // 三轮录音的引导词
  const ENROLL_PROMPTS = [
    { cn: '今天天气不错，同学们早上好', en: 'Good morning, nice weather today' },
    { cn: '一、二、三、四、五、六、七、八、九、十', en: 'One two three four five six seven eight nine ten' },
    { cn: '请说出你的名字和班级', en: 'Please say your name and class' },
  ];

  // ── 验证状态 ─────────────────────────────────────────────────────────────
  const [verifyState, setVerifyState] = useState<'idle' | 'recording' | 'processing'>('idle');
  const [verifySecs, setVerifySecs] = useState(0);
  const [verifyResult, setVerifyResult] = useState<{
    matched: boolean; best_match: string | null; confidence: number; all_scores: Record<string, number>
  } | null>(null);

  // ── 咳嗽检测状态 ─────────────────────────────────────────────────────────
  const [coughState, setCoughState] = useState<'idle' | 'recording' | 'processing' | 'error'>('idle');
  const [coughSecs, setCoughSecs] = useState(0);
  const [coughResult, setCoughResult] = useState<{
    probability: number; is_cough: boolean; label: string; confidence: string
  } | null>(null);
  const [coughError, setCoughError] = useState<string>('');
  const [coughDetectorReady, setCoughDetectorReady] = useState<boolean | null>(null); // null = checking

  // ── 内部录音 refs（各自独立，互不干扰）──────────────────────────────────
  const enrollMrRef  = useRef<MediaRecorder | null>(null);
  const verifyMrRef  = useRef<MediaRecorder | null>(null);
  const coughMrRef   = useRef<MediaRecorder | null>(null);
  const enrollTimer  = useRef<ReturnType<typeof setInterval> | null>(null);
  const verifyTimer  = useRef<ReturnType<typeof setInterval> | null>(null);
  const coughTimer   = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── 切换学生时重置录入状态 ───────────────────────────────────────────────
  useEffect(() => {
    setEnrollState('idle');
    setEnrollSecs(0);
    setEnrollMsg('');
    setEnrollRound(0);
  }, [selectedStudent]);

  // ── 启动时获取已录入名单 ─────────────────────────────────────────────────
  useEffect(() => {
    fetch(`${FLUGUARD_API}/api/voiceprint/profiles`)
      .then(r => r.json())
      .then(d => setEnrolledNames(d.profiles ?? []))
      .catch(() => {});
  }, []);

  // ── 轮询 cough_detector 加载状态，加载完成后停止 ─────────────────────────
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const r = await fetch(`${FLUGUARD_API}/api/health`);
        const d = await r.json();
        if (!cancelled) {
          setCoughDetectorReady(!!d.cough_detector_ready);
          if (!d.cough_detector_ready) {
            // 还没好，2秒后再查
            setTimeout(poll, 2000);
          }
        }
      } catch {
        if (!cancelled) setTimeout(poll, 3000);
      }
    };
    poll();
    return () => { cancelled = true; };
  }, []);

  // ── WAV 编码器：Float32 PCM → 标准 16-bit WAV Blob（无需 ffmpeg）──────
  const encodeWAV = (samples: Float32Array, sampleRate: number): Blob => {
    const numChannels = 1;
    const bitDepth    = 16;
    const byteRate    = sampleRate * numChannels * bitDepth / 8;
    const blockAlign  = numChannels * bitDepth / 8;
    const dataLen     = samples.length * blockAlign;
    const buf         = new ArrayBuffer(44 + dataLen);
    const v           = new DataView(buf);
    const w           = (off: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };

    w(0,  'RIFF'); v.setUint32( 4, 36 + dataLen, true);
    w(8,  'WAVE');
    w(12, 'fmt '); v.setUint32(16, 16, true);
    v.setUint16(20, 1, true);              // PCM
    v.setUint16(22, numChannels, true);
    v.setUint32(24, sampleRate, true);
    v.setUint32(28, byteRate, true);
    v.setUint16(32, blockAlign, true);
    v.setUint16(34, bitDepth, true);
    w(36, 'data'); v.setUint32(40, dataLen, true);

    let off = 44;
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      off += 2;
    }
    return new Blob([buf], { type: 'audio/wav' });
  };

  // ── 录音辅助：AudioContext PCM 录制 → WAV（不依赖 ffmpeg）─────────────
  // mrRef 在此方案中存储一个 "停止句柄" 对象（兼容 stopRecord 调用）
  const record = async (
    mrRef: React.MutableRefObject<any>,
    timerRef: React.MutableRefObject<ReturnType<typeof setInterval> | null>,
    maxSecs: number,
    onTick: (s: number) => void,
    onDone: (blob: Blob) => void,
    onErr:  (msg: string) => void
  ) => {
    try {
      const SR     = 16000;
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: SR, channelCount: 1 } });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: SR }) as AudioContext;
      const source   = audioCtx.createMediaStreamSource(stream);
      // ScriptProcessor is deprecated but still widely supported; works for demo
      const proc     = audioCtx.createScriptProcessor(4096, 1, 1);
      const pcmChunks: Float32Array[] = [];

      proc.onaudioprocess = (e: AudioProcessingEvent) => {
        pcmChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      };
      source.connect(proc);
      proc.connect(audioCtx.destination);

      const stop = () => {
        proc.disconnect(); source.disconnect();
        stream.getTracks().forEach(t => t.stop());
        audioCtx.close();
        // 拼合所有 PCM 块 / merge chunks
        const total   = pcmChunks.reduce((a, b) => a + b.length, 0);
        const merged  = new Float32Array(total);
        let offset    = 0;
        for (const c of pcmChunks) { merged.set(c, offset); offset += c.length; }
        onDone(encodeWAV(merged, SR));
      };

      // 存储停止句柄，使 stopRecord 可调用
      mrRef.current = { state: 'recording', stop };

      let secs = 0;
      timerRef.current = setInterval(() => {
        secs += 1;
        onTick(secs);
        if (secs >= maxSecs) stopRecord(mrRef, timerRef);
      }, 1000);
    } catch (e: any) {
      onErr(e?.message?.includes('Permission') ? '请允许麦克风权限 / Allow microphone' : `录音失败: ${e?.message}`);
    }
  };

  const stopRecord = (
    mrRef: React.MutableRefObject<any>,
    timerRef: React.MutableRefObject<ReturnType<typeof setInterval> | null>
  ) => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    if (mrRef.current?.state === 'recording') { mrRef.current.stop(); mrRef.current = null; }
  };

  // ── 录入（三轮） ──────────────────────────────────────────────────────────
  const TOTAL_ROUNDS = 3;

  const startEnroll = () => {
    const student = students.find(s => s.id === selectedStudent);
    if (!student) return;
    setEnrollState('recording');
    setEnrollSecs(0);
    setEnrollMsg('');

    record(
      enrollMrRef, enrollTimer, 5,
      (s) => setEnrollSecs(s),
      async (blob) => {
        const currentRound = enrollRound; // capture before async
        setEnrollState('uploading');
        setEnrollMsg(`正在提取第 ${currentRound + 1} 段声纹... / Extracting sample ${currentRound + 1}...`);
        try {
          const form = new FormData();
          form.append('name', student.name);
          form.append('audio', blob, 'enroll.wav');
          const resp = await fetch(`${FLUGUARD_API}/api/voiceprint/enroll`, { method: 'POST', body: form });
          if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
          const data = await resp.json();
          setEnrolledNames(data.enrolled_names ?? []);

          const nextRound = currentRound + 1;
          setEnrollRound(nextRound);

          if (nextRound >= TOTAL_ROUNDS) {
            // 三轮全部完成
            setStudents(prev => prev.map(s => s.id === selectedStudent ? { ...s, voiceprintEnrolled: true } : s));
            setEnrollState('done');
            setEnrollMsg('声纹校准完成！精度已提升 / Calibration complete ✓');
          } else {
            // 继续下一轮
            setEnrollState('idle');
            setEnrollMsg(`第 ${nextRound} 段完成，请继续录第 ${nextRound + 1} 段 / Round ${nextRound} done, continue`);
          }
        } catch (e: any) {
          setEnrollState('error');
          setEnrollMsg(`录入失败 / Failed: ${(e as Error).message}`);
        }
      },
      (msg) => { setEnrollState('error'); setEnrollMsg(msg); }
    );
  };

  const resetEnroll = () => {
    setEnrollRound(0);
    setEnrollState('idle');
    setEnrollMsg('');
  };

  const stopEnroll = () => stopRecord(enrollMrRef, enrollTimer);

  // ── 验证 ─────────────────────────────────────────────────────────────────
  const startVerify = () => {
    setVerifyState('recording');
    setVerifySecs(0);
    setVerifyResult(null);

    record(
      verifyMrRef, verifyTimer, 4,
      (s) => setVerifySecs(s),
      async (blob) => {
        setVerifyState('processing');
        try {
          const form = new FormData();
          form.append('audio', blob, 'verify.wav');
          const resp = await fetch(`${FLUGUARD_API}/api/voiceprint/verify`, { method: 'POST', body: form });
          if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
          setVerifyResult(await resp.json());
        } catch { setVerifyResult(null); }
        finally { setVerifyState('idle'); }
      },
      () => setVerifyState('idle')
    );
  };

  const stopVerify = () => stopRecord(verifyMrRef, verifyTimer);

  // ── 咳嗽检测 ─────────────────────────────────────────────────────────────
  const startCough = () => {
    setCoughState('recording');
    setCoughSecs(0);
    setCoughResult(null);
    setCoughError('');

    record(
      coughMrRef, coughTimer, 3,
      (s) => setCoughSecs(s),
      async (blob) => {
        setCoughState('processing');
        try {
          const form = new FormData();
          form.append('audio', blob, 'cough.wav');
          const resp = await fetch(`${FLUGUARD_API}/api/cough/detect`, { method: 'POST', body: form });
          if (!resp.ok) {
            const detail = await resp.json().catch(() => ({}));
            throw new Error(detail?.detail ?? `HTTP ${resp.status}`);
          }
          setCoughResult(await resp.json());
          setCoughState('idle');
        } catch (e: any) {
          setCoughError((e as Error).message ?? '检测失败');
          setCoughState('error');
        }
      },
      (msg) => { setCoughError(msg); setCoughState('error'); }
    );
  };

  const stopCough = () => stopRecord(coughMrRef, coughTimer);
  const resetCough = () => { setCoughState('idle'); setCoughResult(null); setCoughError(''); };

  const selectedStudentName = students.find(s => s.id === selectedStudent)?.name ?? '';

  return (
    <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="max-w-5xl mx-auto space-y-8">
      <header className="text-center space-y-3">
        <h2 className="text-4xl font-black tracking-tight text-slate-900">
          Audio Intelligence
          <span className="block text-xl opacity-40 font-bold mt-1">声纹录入 · 咳嗽检测</span>
        </h2>
        <p className="text-slate-400 text-sm">
          resemblyzer GE2E 声纹 · YAMNet fine-tuned 咳嗽检测 (AUC 99.6% · F1 97.4%)
        </p>
      </header>

      {/* ══ 声纹录入 ══════════════════════════════════════════════════════════ */}
      <div className="bg-white rounded-[40px] border border-slate-100 shadow-sm overflow-hidden">
        <div className="px-8 pt-8 pb-4">
          <h3 className="text-base font-black text-slate-900 flex items-center gap-2">
            <Fingerprint className="w-5 h-5 text-blue-600" />
            声纹录入 / Voiceprint Enrollment
          </h3>
        </div>
        <div className="grid grid-cols-5 divide-x divide-slate-100">

          {/* 左：学生列表 */}
          <div className="col-span-2 p-6 space-y-2 max-h-72 overflow-y-auto">
            {students.map(student => {
              const isSelected  = student.id === selectedStudent;
              const isEnrolled  = enrolledNames.includes(student.name);
              return (
                <button
                  key={student.id}
                  onClick={() => setSelectedStudent(student.id)}
                  className={cn(
                    "w-full px-4 py-3 rounded-2xl border text-left transition-all flex items-center justify-between gap-2",
                    isSelected
                      ? "border-blue-500 bg-blue-50 shadow-sm"
                      : "border-slate-100 bg-white hover:border-blue-200 hover:bg-slate-50"
                  )}
                >
                  <div className="min-w-0">
                    <div className={cn("font-bold text-sm truncate", isSelected ? "text-blue-600" : "text-slate-800")}>
                      {student.name}
                    </div>
                    <div className="text-[10px] text-slate-400 mt-0.5">Grade 1 · Class 1</div>
                  </div>
                  {isEnrolled && (
                    <span className="shrink-0 text-[10px] font-black text-emerald-600 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full">
                      Enrolled / 已录入
                    </span>
                  )}
                </button>
              );
            })}
          </div>

          {/* 右：录入操作区 */}
          <div className="col-span-3 p-8 flex flex-col items-center justify-center gap-5">

            {/* 三轮进度指示器 */}
            <div className="flex items-center gap-3">
              {[0, 1, 2].map(i => (
                <div key={i} className="flex flex-col items-center gap-1">
                  <div className={cn(
                    "w-8 h-8 rounded-full flex items-center justify-center text-xs font-black transition-all",
                    i < enrollRound
                      ? "bg-emerald-500 text-white"
                      : i === enrollRound && enrollState !== 'done'
                        ? "bg-blue-600 text-white ring-4 ring-blue-100"
                        : enrollState === 'done'
                          ? "bg-emerald-500 text-white"
                          : "bg-slate-100 text-slate-400"
                  )}>
                    {i < enrollRound || enrollState === 'done' ? '✓' : i + 1}
                  </div>
                  <div className="text-[9px] text-slate-400 font-bold">R{i + 1} / 第{i + 1}段</div>
                </div>
              ))}
            </div>

            {/* 引导文字 */}
            {enrollState !== 'done' && (
              <div className="w-full bg-blue-50 border border-blue-100 rounded-3xl px-6 py-4 text-center">
                <div className="text-[10px] font-black text-blue-500 uppercase tracking-widest mb-1">
                  Round {enrollRound + 1} / {TOTAL_ROUNDS} · Enrolling <span className="text-blue-700">{selectedStudentName.split(' ')[0]}</span>
                  <span className="block text-[9px] text-blue-400 font-bold mt-0.5 normal-case tracking-normal">
                    第 {enrollRound + 1} / {TOTAL_ROUNDS} 段 · 为 {selectedStudentName.split(' ')[0]} 录入
                  </span>
                </div>
                <div className="font-bold text-slate-800 text-sm mt-1">
                  "{ENROLL_PROMPTS[Math.min(enrollRound, 2)].cn}"
                </div>
                <div className="text-[10px] text-slate-400 mt-1">
                  {ENROLL_PROMPTS[Math.min(enrollRound, 2)].en}
                </div>
                <div className="text-[10px] text-slate-300 mt-2">Read for 5s · Only a 256-dim feature vector is stored, no raw audio<br/><span className="text-slate-300/80">朗读 5 秒 · 仅存储 256 维特征向量，不保存原始音频</span></div>
              </div>
            )}

            {enrollState === 'done' && (
              <div className="w-full bg-emerald-50 border border-emerald-200 rounded-3xl px-6 py-5 text-center">
                <div className="text-emerald-700 font-black text-base">Voiceprint Calibrated ✓ / 声纹校准完成</div>
                <div className="text-[11px] text-emerald-600 mt-1">3 段样本已融合，识别精度已提升 / 3 samples merged</div>
              </div>
            )}

            {/* 麦克风按钮 */}
            <motion.div
              animate={enrollState === 'recording' ? { scale: [1, 1.12, 1] } : { scale: 1 }}
              transition={enrollState === 'recording' ? { repeat: Infinity, duration: 1 } : { duration: 0.2 }}
              className={cn(
                "w-20 h-20 rounded-full flex items-center justify-center shadow-xl",
                enrollState === 'recording' ? "bg-red-500"
                : enrollState === 'uploading' ? "bg-amber-400"
                : enrollState === 'done'      ? "bg-emerald-500"
                : enrollState === 'error'     ? "bg-rose-400"
                : "bg-blue-600"
              )}
            >
              {enrollState === 'done'
                ? <CheckCircle2 className="w-9 h-9 text-white" />
                : <Mic className="w-9 h-9 text-white" />}
            </motion.div>

            {/* 进度条（录音中） */}
            {enrollState === 'recording' && (
              <div className="w-full max-w-xs space-y-1 text-center">
                <div className="text-red-500 font-black text-sm animate-pulse">REC · {enrollSecs}s / 5s</div>
                <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-red-500 rounded-full transition-all" style={{ width: `${(enrollSecs / 5) * 100}%` }} />
                </div>
              </div>
            )}

            {/* 状态提示 */}
            {enrollMsg && enrollState !== 'done' && (
              <div className={cn("text-sm font-bold px-4 py-2 rounded-2xl text-center",
                enrollState === 'uploading' ? "text-amber-700 bg-amber-50"
                : enrollState === 'error'   ? "text-red-600 bg-red-50"
                : "text-blue-700 bg-blue-50"
              )}>
                {enrollMsg}
              </div>
            )}

            {/* 操作按钮 */}
            <div className="flex gap-3">
              {enrollState !== 'recording' && enrollState !== 'uploading' && enrollState !== 'done' && (
                <button
                  onClick={startEnroll}
                  className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-2xl font-bold text-sm hover:bg-blue-700 active:scale-95 transition-all shadow-lg shadow-blue-100"
                >
                  <Mic className="w-4 h-4" />
                  {enrollRound === 0 ? 'Start / 开始录入' : `Record Round ${enrollRound + 1} / 录第 ${enrollRound + 1} 段`}
                </button>
              )}
              {enrollState === 'done' && (
                <button
                  onClick={resetEnroll}
                  className="flex items-center gap-2 px-6 py-3 bg-slate-100 text-slate-700 rounded-2xl font-bold text-sm hover:bg-slate-200 active:scale-95 transition-all"
                >
                  重新录入 / Re-enroll
                </button>
              )}
              {enrollState === 'recording' && (
                <button
                  onClick={stopEnroll}
                  className="flex items-center gap-2 px-6 py-3 bg-slate-800 text-white rounded-2xl font-bold text-sm hover:bg-slate-900 active:scale-95 transition-all"
                >
                  提前停止 / Stop
                </button>
              )}
              {enrollState === 'uploading' && (
                <div className="flex items-center gap-2 px-6 py-3 bg-amber-100 text-amber-700 rounded-2xl font-bold text-sm animate-pulse">
                  Extracting features... / 提取特征中...
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* ══ 声纹验证 + 咳嗽检测（并排）══════════════════════════════════════ */}
      <div className="grid grid-cols-2 gap-6">

        {/* 声纹验证 */}
        <div className="bg-white p-7 rounded-[32px] border border-slate-100 shadow-sm space-y-5">
          <div>
            <h3 className="font-black text-slate-900 flex items-center gap-2">
              <Fingerprint className="w-5 h-5 text-purple-600" />
              Voiceprint Verify / 声纹验证
            </h3>
            {enrolledNames.length > 0
              ? <p className="text-xs text-slate-400 mt-1">Enrolled / 已录入：{enrolledNames.join(' · ')}</p>
              : <p className="text-xs text-amber-500 mt-1">⚠ Please enroll at least one student above first / 请先在上方录入至少一名学生的声纹</p>
            }
          </div>

          <button
            onClick={verifyState === 'recording' ? stopVerify : startVerify}
            disabled={enrolledNames.length === 0 && verifyState === 'idle'}
            className={cn(
              "w-full flex items-center justify-center gap-2 py-3 rounded-2xl font-bold text-sm transition-all active:scale-95",
              verifyState === 'recording' ? "bg-red-500 text-white animate-pulse"
              : verifyState === 'processing' ? "bg-slate-200 text-slate-500 cursor-wait"
              : enrolledNames.length === 0 ? "bg-slate-100 text-slate-400 cursor-not-allowed"
              : "bg-purple-600 text-white hover:bg-purple-700 shadow-lg shadow-purple-100"
            )}
          >
            <Mic className="w-4 h-4" />
            {verifyState === 'recording'   ? `Recording ${verifySecs}s / 4s · 录音中`
            : verifyState === 'processing' ? 'Identifying... / 识别中...'
            : 'Verify / 开始验证'}
          </button>

          {verifyResult && (
            <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
              className={cn("p-4 rounded-2xl border", verifyResult.matched ? "bg-emerald-50 border-emerald-200" : "bg-slate-50 border-slate-200")}
            >
              <div className={cn("font-black text-base", verifyResult.matched ? "text-emerald-700" : "text-slate-600")}>
                {verifyResult.matched ? `✓ ${verifyResult.best_match}` : '✗ No match / 未识别到'}
              </div>
              <div className="text-xs text-slate-500 mt-1 space-y-0.5">
                {Object.entries(verifyResult.all_scores).map(([n, s]) => (
                  <div key={n} className="flex justify-between">
                    <span>{n}</span>
                    <span className="font-bold">{(s * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </div>

        {/* 咳嗽检测 */}
        <div className="bg-white p-7 rounded-[32px] border border-slate-100 shadow-sm space-y-5">
          <div>
            <h3 className="font-black text-slate-900 flex items-center gap-2">
              <Activity className="w-5 h-5 text-rose-600" />
              Cough Detect / 咳嗽检测
            </h3>
            <p className="text-xs text-slate-400 mt-1">YAMNet + fine-tuned · AUC 99.6%</p>
          </div>

          {/* 主按钮 — 模型加载中时显示等待提示 */}
          {coughDetectorReady === null && (
            <div className="w-full flex items-center justify-center gap-2 py-3 rounded-2xl bg-slate-100 text-slate-400 text-sm font-medium">
              <span className="animate-spin inline-block w-4 h-4 border-2 border-slate-300 border-t-slate-500 rounded-full" />
              Model loading... / 模型加载中
            </div>
          )}
          {coughDetectorReady === false && (
            <div className="w-full flex items-center justify-center gap-2 py-3 rounded-2xl bg-amber-50 border border-amber-200 text-amber-700 text-sm font-medium">
              ⏳ Starting up YAMNet model, please wait… / YAMNet 模型启动中，请稍候…
            </div>
          )}
          {coughDetectorReady === true && (coughState === 'idle' || coughState === 'recording' || coughState === 'processing') && (
            <button
              onClick={coughState === 'recording' ? stopCough : startCough}
              disabled={coughState === 'processing'}
              className={cn(
                "w-full flex items-center justify-center gap-2 py-3 rounded-2xl font-bold text-sm transition-all active:scale-95",
                coughState === 'recording'   ? "bg-red-500 text-white"
                : coughState === 'processing' ? "bg-slate-200 text-slate-500 cursor-wait"
                : "bg-rose-600 text-white hover:bg-rose-700 shadow-lg shadow-rose-100"
              )}
            >
              <Mic className="w-4 h-4" />
              {coughState === 'recording'   ? `● Recording ${coughSecs}s / 3s — tap to stop · 点击提前停止`
              : coughState === 'processing' ? '🔍 AI analyzing... / AI 分析中'
              : '🎤 Record & Detect / 录音检测'}
            </button>
          )}

          {/* 录音进度条 */}
          {coughState === 'recording' && (
            <div className="space-y-1">
              <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-red-500 rounded-full transition-all duration-1000"
                  style={{ width: `${(coughSecs / 3) * 100}%` }}
                />
              </div>
              <p className="text-[10px] text-slate-400 text-center">Cough into the mic — auto-stops after 3 seconds / 对准麦克风咳嗽，3 秒后自动停止</p>
            </div>
          )}

          {/* 检测结果 */}
          {coughResult && coughState === 'idle' && (
            <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
              className={cn("p-5 rounded-2xl border text-center space-y-2",
                coughResult.is_cough ? "bg-red-50 border-red-200" : "bg-emerald-50 border-emerald-200"
              )}
            >
              <div className={cn("text-2xl font-black", coughResult.is_cough ? "text-red-600" : "text-emerald-600")}>
                {coughResult.is_cough ? '🔴 COUGH DETECTED / 检测到咳嗽' : '🟢 NO COUGH / 未检测到咳嗽'}
              </div>
              <div className="text-xs text-slate-500">
                Probability {(coughResult.probability * 100).toFixed(1)}% · Confidence {coughResult.confidence}
                <span className="block text-[10px] text-slate-400">概率 / 置信度</span>
              </div>
              <button
                onClick={resetCough}
                className="text-xs text-slate-400 underline hover:text-slate-600 mt-1"
              >
                再次检测 / Test Again
              </button>
            </motion.div>
          )}

          {/* 错误提示 */}
          {coughState === 'error' && (
            <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
              className="p-4 rounded-2xl border bg-amber-50 border-amber-200 text-center space-y-2"
            >
              <div className="text-amber-700 font-bold text-sm">⚠️ 检测失败 / Detection Failed</div>
              <div className="text-xs text-amber-600">{coughError}</div>
              <button
                onClick={resetCough}
                className="text-xs text-amber-700 underline hover:text-amber-900"
              >
                重试 / Retry
              </button>
            </motion.div>
          )}

          <p className="text-[11px] text-slate-400">Training Set / 训练集：CoughVid v3 × 1000 + ESC-50 × 1000 + LibriSpeech × 500</p>
        </div>
      </div>

      {/* ══ 底部说明 ═════════════════════════════════════════════════════════ */}
      <div className="bg-slate-900 p-7 rounded-[32px] text-white flex items-center justify-between">
        <div>
          <div className="font-black text-base mb-1">Privacy-First / 隐私优先</div>
          <p className="text-slate-400 text-xs max-w-lg">
            Only a 256-dim voiceprint vector is stored — no raw audio. Cough detection returns a probability only. All inference runs locally; nothing is uploaded to the cloud.
            <span className="block mt-1 text-slate-500">声纹仅存 256 维向量，不保存原始音频。咳嗽检测仅返回概率值。所有推理在本机完成，数据不上传云端。</span>
          </p>
        </div>
        <Fingerprint className="w-12 h-12 text-slate-700 shrink-0" />
      </div>
    </motion.div>
  );
}

function HealthGuide({ user, classrooms }: { user: UserProfile, classrooms: Classroom[] }) {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isTyping, setIsTyping] = useState(false);

  // 模拟调用 Gemma 进行健康咨询
  const handleSend = async () => {
    if (!query.trim()) return;
    
    const userMsg = { role: 'user' as const, content: query };
    setMessages(prev => [...prev, userMsg]);
    setQuery('');
    setIsTyping(true);

    try {
      let rolePrompt = SYSTEM_PROMPTS.REPORT;
      if (user.role === 'principal') rolePrompt = SYSTEM_PROMPTS.PRINCIPAL;
      else if (user.role === 'teacher') rolePrompt = SYSTEM_PROMPTS.TEACHER;
      else if (user.role === 'parent') rolePrompt = SYSTEM_PROMPTS.PARENT;
      else if (user.role === 'student') rolePrompt = SYSTEM_PROMPTS.STUDENT;
      else if (user.role === 'bureau') rolePrompt = SYSTEM_PROMPTS.BUREAU;

      const systemPrompt = rolePrompt + "\n\n" + SYSTEM_PROMPTS.SAFETY;

      const result = await callFluGuardAI({
        role: user.role,
        message: `${query}\n\nIMPORTANT: Your response MUST be bilingual. Provide English first, then Chinese below it.`,
        system_prompt: systemPrompt,
      });
      setMessages(prev => [...prev, { role: 'assistant', content: result.text || 'Sorry, I cannot answer this question right now. \n 抱歉，我暂时无法回答这个问题。' }]);
    } catch (error) {
      console.error('Health guide failed:', error);
    } finally {
      setIsTyping(false);
    }
  };

  const getWelcomeMessage = () => {
    switch(user.role) {
      case 'principal':
        return {
          title: "Hello, Principal. I'm your Health Decision Assistant",
          titleCn: "校长您好，我是您的校园健康决策助手",
          desc: "I can provide data-driven analysis on school flu trends and regulatory compliance advice.\n您可以询问关于全校流感趋势分析、卫生法规合规性建议及应急响应方案。",
          icon: Building2
        };
      case 'bureau':
        return {
          title: "Hello, Bureau Official. I'm your Regional Health Strategist",
          titleCn: "领导您好，我是您的区域健康战略助手",
          desc: "I can provide macro-level analysis on regional flu clusters and policy recommendations.\n您可以询问关于区域流感集群分析、资源调度建议及政策性预警方案。",
          icon: ShieldCheck
        };
      case 'teacher':
        return {
          title: "Hello, Teacher. I'm your Classroom Health Assistant",
          titleCn: "老师您好，我是您的班级健康管理助手",
          desc: "I can help with classroom flu prevention, student health monitoring, and teaching continuity advice.\n您可以询问关于班级流感预防、学生健康监测及教学连续性保障建议。",
          icon: Users
        };
      case 'parent':
        return {
          title: "Hello, Parent. I'm your Family Health Assistant",
          titleCn: "家长您好，我是您的家庭健康关怀助手",
          desc: "I can provide caring advice on flu prevention and child care. (No medication prescriptions)\n您可以询问关于孩子流感预防、日常护理及就医引导建议（不提供药物处方）。",
          icon: Heart
        };
      case 'student':
        return {
          title: "Hi! I'm Xiao Wei, your Health Big Brother",
          titleCn: "嗨！我是小卫，你的健康大哥哥",
          desc: "Ask me anything about staying healthy! Remember to tell your parents and teacher if you feel unwell.\n关于健康的问题都可以问我哦！如果不舒服记得一定要告诉爸爸妈妈和老师。",
          icon: Smile
        };
      default:
        return {
          title: "Hello, I'm your AI School Doctor",
          titleCn: "您好，我是您的 AI 校医",
          desc: "You can ask me anything about flu prevention or health advice.\n您可以询问我关于流感预防或健康建议的任何问题。",
          icon: Stethoscope
        };
    }
  };

  const welcome = getWelcomeMessage();

  return (
    <div className="space-y-12 max-w-[1200px] mx-auto h-[calc(100vh-160px)] flex flex-col">
      <header className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 shrink-0">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-emerald-600 font-black text-xs uppercase tracking-widest">
            <div className="w-2 h-2 bg-emerald-600 rounded-full animate-pulse" />
            AI Health Assistant Active
          </div>
          <h2 className="text-5xl font-black text-slate-900 tracking-tight">
            {user.role === 'student' ? 'Health Guide' : 'AI Health Guide'}<br />
            <span className="text-slate-400 font-serif italic text-4xl">{welcome.titleCn.split('，')[1] || welcome.titleCn}</span>
          </h2>
          <p className="text-slate-500 font-medium">{welcome.title.split('.')[0]}.<br/><span className="text-xs opacity-60">{welcome.desc.split('\n')[1]}</span></p>
        </div>
      </header>

      <div className="flex-1 bento-card flex flex-col overflow-hidden p-0">
        <div className="flex-1 overflow-y-auto p-10 space-y-8">
          {messages.length === 0 && (
            <div className="min-h-full flex flex-col items-center text-center space-y-6 max-w-md mx-auto py-8">
              <div className="w-20 h-20 bg-emerald-50 rounded-[32px] flex items-center justify-center text-emerald-600 shadow-xl shadow-emerald-50 shrink-0">
                <welcome.icon className="w-10 h-10" />
              </div>
              <div className="space-y-3 w-full">
                <h3 className="text-xl font-black text-slate-900 leading-tight break-words">{welcome.title}</h3>
                <span className="block text-sm opacity-50 font-bold break-words">{welcome.titleCn}</span>
                <p className="text-slate-500 font-medium leading-relaxed whitespace-pre-wrap text-sm">
                  {welcome.desc}
                </p>
              </div>
              <div className="grid grid-cols-1 gap-3 w-full">
                {[
                  { en: 'How to prevent flu?', zh: '如何预防甲流？' },
                  { en: 'Cough in class?', zh: '班级咳嗽多怎么办？' },
                  { en: 'Need to see a doctor?', zh: '孩子轻微咳嗽需要就医吗？' }
                ].map(q => (
                  <button 
                    key={q.en}
                    onClick={() => { setQuery(q.en); handleSend(); }}
                    className="p-4 bg-slate-50 rounded-2xl text-sm font-bold text-slate-600 hover:bg-slate-100 transition-all border border-slate-100 flex flex-col items-center"
                  >
                    <span>{q.en}</span>
                    <span className="text-[10px] opacity-50">{q.zh}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
          
          {messages.map((msg, i) => (
            <motion.div 
              key={i}
              initial={{ opacity: 0, y: 10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              className={cn(
                "flex gap-4 max-w-[85%]",
                msg.role === 'user' ? "ml-auto flex-row-reverse" : "mr-auto"
              )}
            >
              <div className={cn(
                "w-10 h-10 rounded-2xl flex items-center justify-center shrink-0 shadow-lg",
                msg.role === 'user' ? "bg-slate-900 text-white" : "bg-emerald-600 text-white"
              )}>
                {msg.role === 'user' ? <UserIcon className="w-5 h-5" /> : <BrainCircuit className="w-5 h-5" />}
              </div>
              <div className={cn(
                "p-5 rounded-[28px] text-sm leading-relaxed font-medium shadow-sm border whitespace-pre-wrap",
                msg.role === 'user' 
                  ? "bg-slate-900 text-white border-slate-800 rounded-tr-none" 
                  : "bg-white text-slate-700 border-slate-100 rounded-tl-none"
              )}>
                {msg.content}
              </div>
            </motion.div>
          ))}
          
          {isTyping && (
            <div className="flex gap-4 mr-auto">
              <div className="w-10 h-10 rounded-2xl bg-emerald-600 text-white flex items-center justify-center shrink-0 shadow-lg">
                <BrainCircuit className="w-5 h-5" />
              </div>
              <div className="p-5 bg-white rounded-[28px] rounded-tl-none border border-slate-100 flex gap-1.5 items-center shadow-sm">
                <motion.div 
                  animate={{ scale: [1, 1.2, 1] }} 
                  transition={{ repeat: Infinity, duration: 0.6 }} 
                  className="w-1.5 h-1.5 bg-emerald-400 rounded-full" 
                />
                <motion.div 
                  animate={{ scale: [1, 1.2, 1] }} 
                  transition={{ repeat: Infinity, duration: 0.6, delay: 0.2 }} 
                  className="w-1.5 h-1.5 bg-emerald-400 rounded-full" 
                />
                <motion.div 
                  animate={{ scale: [1, 1.2, 1] }} 
                  transition={{ repeat: Infinity, duration: 0.6, delay: 0.4 }} 
                  className="w-1.5 h-1.5 bg-emerald-400 rounded-full" 
                />
              </div>
            </div>
          )}
        </div>

        <div className="p-8 bg-slate-50 border-t border-slate-100">
          <form 
            onSubmit={(e) => { e.preventDefault(); handleSend(); }}
            className="flex gap-4"
          >
            <input 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="输入您的问题... / Type your question..."
              className="flex-1 bg-white border border-slate-200 rounded-2xl px-6 py-4 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 transition-all shadow-sm"
            />
            <button 
              type="submit"
              disabled={!query.trim() || isTyping}
              className="px-8 py-4 bg-emerald-600 text-white rounded-2xl font-black text-sm hover:bg-emerald-700 transition-all shadow-xl shadow-emerald-100 disabled:opacity-50 flex items-center gap-2"
            >
              <Send className="w-5 h-5" /> 发送 / Send
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

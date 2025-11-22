'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '@/context/AuthContext';
import { authAPI, setAccessToken } from '@/lib/api';
import Header from '@/components/Header';

export default function LoginPage() {
  const router = useRouter();
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const response = await authAPI.login(email, password);
      const { accessToken, user } = response.data;

      // Set token and user
      setAccessToken(accessToken);
      login(accessToken, user);

      // Redirect to home
      router.push('/'); // The home page will now handle displaying recommendations
    } catch (err: any) {
      const status = err.response?.status;
      const errorMessage = err.response?.data?.error || 'Login failed';

      if (status === 404) {
        // No account exists for this email → suggest signup
        setError('NOT_REGISTERED'); // Use a special error code for the modal
      } else if (status === 401) {
        // Account exists but password incorrect
        setError('이메일 또는 비밀번호가 올바르지 않습니다');
      } else {
        setError(errorMessage);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white">
      <Header />

      <div className="container mx-auto px-4 pt-24 flex items-center justify-center min-h-screen">
        <div className="w-full max-w-md">
          <div className="bg-gray-900 rounded-lg p-8 shadow-xl">
            <div className="text-center mb-8">
              <h1 className="text-3xl font-bold mb-2">MovieFlix</h1>
              <p className="text-gray-400">로그인하여 영화를 탐색하세요</p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label htmlFor="email" className="block text-sm font-medium mb-2">
                  이메일
                </label>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded focus:outline-none focus:border-red-500"
                  placeholder="your@email.com"
                />
              </div>

              <div>
                <label htmlFor="password" className="block text-sm font-medium mb-2">
                  비밀번호
                </label>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded focus:outline-none focus:border-red-500"
                  placeholder="••••••••"
                />
              </div>

              {error && error !== 'NOT_REGISTERED' && (
                <div className="bg-red-500/10 border border-red-500 text-red-500 px-4 py-2 rounded text-sm">
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-red-600 text-white py-2 rounded font-medium hover:bg-red-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? '로그인 중...' : '로그인'}
              </button>
            </form>

            <div className="mt-6 text-center text-sm text-gray-400">
              계정이 없으신가요?{' '}
              <Link href="/signup" className="text-red-500 hover:text-red-400">
                회원가입
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Not Registered Modal */}
      {error === 'NOT_REGISTERED' && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm" onClick={() => setError('')}>
          <div className="bg-gray-900 border-2 border-red-500 rounded-lg p-8 max-w-md mx-4 shadow-2xl">
            <div className="text-center">
              {/* Icon */}
              <div className="mb-4 flex justify-center">
                <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center">
                  <svg
                    className="w-10 h-10 text-red-500"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                    />
                  </svg>
                </div>
              </div>

              {/* Message */}
              <h2 className="text-2xl font-bold text-white mb-3">
                계정을 찾을 수 없습니다
              </h2>
              <p className="text-gray-300 mb-6">
                이 이메일로 등록된 계정이 없습니다.<br />
                먼저 회원가입을 해주세요.
              </p>

              {/* Buttons */}
              <div className="flex flex-col gap-3">
                <button
                  onClick={() => router.push('/signup')}
                  className="w-full bg-red-600 text-white py-3 rounded-lg font-medium hover:bg-red-700 transition"
                >
                  회원가입하러 가기
                </button>
                <button
                  onClick={() => setError('')}
                  className="w-full bg-gray-800 text-gray-300 py-3 rounded-lg font-medium hover:bg-gray-700 transition"
                >
                  닫기
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

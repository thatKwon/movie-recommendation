'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '@/context/AuthContext';
import { authAPI, setAccessToken } from '@/lib/api';
import Header from '@/components/Header';

export default function SignupPage() {
  const router = useRouter();
  const { login } = useAuth();
  const [name, setName] = useState(''); // Nickname state
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (password !== confirmPassword) {
      setError('비밀번호가 일치하지 않습니다');
      return;
    }

    if (password.length < 6) {
      setError('비밀번호는 최소 6자 이상이어야 합니다');
      return;
    }

    // Validate Name
    if (!name.trim()) {
      setError('닉네임을 입력해주세요');
      return;
    }

    setLoading(true);

    try {
      // SEND NAME TO API
      const response = await authAPI.signup({ email, password, name });
      const { accessToken, user } = response.data;

      setAccessToken(accessToken);
      login(accessToken, user);

      router.push('/questionnaire');
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.error || 'Signup failed');
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
                <p className="text-gray-400">회원가입하고 영화를 추천받으세요</p>
              </div>

              <form onSubmit={handleSubmit} className="space-y-6">

                {/* NICKNAME INPUT */}
                <div>
                  <label htmlFor="name" className="block text-sm font-medium mb-2">
                    닉네임
                  </label>
                  <input
                      id="name"
                      type="text"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      required
                      className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded focus:outline-none focus:border-red-500"
                      placeholder="닉네임을 입력하세요"
                  />
                </div>

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
                      placeholder="최소 6자"
                  />
                </div>

                <div>
                  <label htmlFor="confirmPassword" className="block text-sm font-medium mb-2">
                    비밀번호 확인
                  </label>
                  <input
                      id="confirmPassword"
                      type="password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      required
                      className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded focus:outline-none focus:border-red-500"
                      placeholder="비밀번호 다시 입력"
                  />
                </div>

                {error && (
                    <div className="bg-red-500/10 border border-red-500 text-red-500 px-4 py-2 rounded text-sm">
                      {error}
                    </div>
                )}

                <button
                    type="submit"
                    disabled={loading}
                    className="w-full bg-red-600 text-white py-2 rounded font-medium hover:bg-red-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? '가입 중...' : '회원가입'}
                </button>
              </form>

              <div className="mt-6 text-center text-sm text-gray-400">
                이미 계정이 있으신가요?{' '}
                <Link href="/login" className="text-red-500 hover:text-red-400">
                  로그인
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
  );
}
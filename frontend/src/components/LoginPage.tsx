import { useState } from 'react';
import { Film, LogIn } from 'lucide-react';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { Label } from './ui/label';

interface LoginPageProps {
  onLogin: () => void;
}

export function LoginPage({ onLogin }: LoginPageProps) {
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    // Simple validation - in a real app, this would check against a backend
    if (!userId.trim() || !password.trim()) {
      setError('아이디와 비밀번호를 입력해주세요');
      return;
    }

    // For demo purposes, accept any non-empty credentials
    onLogin();
  };

  return (
    <div className="min-h-screen bg-black flex items-center justify-center px-4">
      <div className="w-full max-w-md space-y-8">
        {/* Logo & Title */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center gap-2">
            <Film className="h-12 w-12 text-red-600" />
            <h1 className="text-4xl text-white">MovieFlix</h1>
          </div>
          <p className="text-zinc-400">당신을 위한 완벽한 영화 추천</p>
        </div>

        {/* Login Form */}
        <div className="bg-zinc-900 rounded-lg p-8 space-y-6 border border-zinc-800">
          <h2 className="text-2xl text-white text-center">로그인</h2>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="userId" className="text-zinc-300">아이디</Label>
              <Input
                id="userId"
                type="text"
                placeholder="아이디를 입력하세요"
                value={userId}
                onChange={(e) => {
                  setUserId(e.target.value);
                  setError('');
                }}
                className="bg-zinc-800 border-zinc-700 text-white placeholder:text-zinc-500 focus:border-red-600"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password" className="text-zinc-300">비밀번호</Label>
              <Input
                id="password"
                type="password"
                placeholder="비밀번호를 입력하세요"
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  setError('');
                }}
                className="bg-zinc-800 border-zinc-700 text-white placeholder:text-zinc-500 focus:border-red-600"
              />
            </div>

            {error && (
              <div className="text-red-500 text-sm text-center">
                {error}
              </div>
            )}

            <Button
              type="submit"
              className="w-full h-12 bg-red-600 hover:bg-red-700 text-white"
            >
              <LogIn className="h-5 w-5 mr-2" />
              로그인
            </Button>
          </form>

          <div className="text-center space-y-2">
            <p className="text-sm text-zinc-500">
              현재는 아무 값이나 입력해도 로그인 가능
            </p>
          </div>
        </div>

        {/* Footer */}
        <p className="text-center text-sm text-zinc-600">
          © 2024 MovieFlix. All rights reserved.
        </p>
      </div>
    </div>
  );
}

'use client';

import Link from 'next/link';
import { useAuth } from '@/context/AuthContext';

export default function Header() {
  const { isAuthenticated, user } = useAuth();

  const clearSearchState = () => {
    try {
      sessionStorage.removeItem('search_state_v1');
    } catch (e) {
      // Ignore errors
    }
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-black/90 backdrop-blur-sm border-b border-gray-800">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" onClick={clearSearchState} className="flex items-center gap-2">
          <div className="w-8 h-8 bg-red-600 flex items-center justify-center rounded">
            <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 20 20">
              <rect x="2" y="4" width="16" height="12" rx="1" />
              <rect x="4" y="7" width="5" height="3" fill="black" />
              <rect x="11" y="7" width="5" height="3" fill="black" />
            </svg>
          </div>
          <span className="text-white text-xl font-bold">MovieFlix</span>
        </Link>

        {/* Navigation */}
        <nav className="flex items-center gap-6">
          <Link href="/" onClick={clearSearchState} className="flex items-center gap-2 text-white hover:text-red-500 transition">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
            </svg>
            <span>홈</span>
          </Link>

          <Link href="/search" className="flex items-center gap-2 text-white hover:text-red-500 transition">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <span>검색</span>
          </Link>

          {isAuthenticated ? (
            <Link href="/user" onClick={clearSearchState} className="flex items-center gap-2 text-white hover:text-red-500 transition">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
              <span>마이페이지</span>
            </Link>
          ) : (
            <Link href="/login" className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition">
              로그인
            </Link>
          )}
        </nav>
      </div>
    </header>
  );
}

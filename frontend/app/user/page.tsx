'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import Link from 'next/link';
import { useAuth } from '@/context/AuthContext';
import { likesAPI } from '@/lib/api';
import { useTheme } from '@/context/ThemeContext';
import Header from '@/components/Header';

// Define the structure of a Like item coming from the API
type LikeItem = {
  id: string;
  targetId: string;
  targetType: 'Movie' | 'Actor' | 'Director';
  target: {
    _id: string;
    title?: string;
    name?: string;
    posterUrl?: string;
    profileUrl?: string;
    year?: number;
  };
};

export default function UserPage() {
  const { user, isAuthenticated, logout } = useAuth();
  const { darkMode, toggleDarkMode } = useTheme();
  const router = useRouter();

  // --- State ---
  const [likes, setLikes] = useState<LikeItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'Movie' | 'Actor' | 'Director'>('Movie');

  // --- Modal State ---
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [itemToDelete, setItemToDelete] = useState<{ type: string; id: string } | null>(null);
  const [doNotShowAgain, setDoNotShowAgain] = useState(false); // Checkbox state

  // 1. Fetch Data
  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/login');
      return;
    }

    const fetchLikes = async () => {
      try {
        setLoading(true);
        const response = await likesAPI.getAll();
        setLikes(response.data);
      } catch (error) {
        console.error('Failed to fetch likes:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchLikes();
  }, [isAuthenticated, router]);

  // --- Logic: Execute the actual delete ---
  const executeDelete = async (type: string, id: string) => {
    try {
      // Optimistic UI Update
      setLikes((prev) => prev.filter((item) => item.target._id !== id));

      // API Call
      await likesAPI.deleteByTarget(type, id);
    } catch (error) {
      console.error('Error unliking item:', error);
      alert('오류가 발생했습니다.');
      // You might want to re-fetch data here if the API call fails
    }
  };

  // 2. Trigger Logic (Check Preference or Open Modal)
  const initiateUnlike = (targetType: string, targetId: string) => {
    const skipConfirmation = localStorage.getItem('skip_unlike_confirmation');

    if (skipConfirmation === 'true') {
      // If preference is saved, skip modal and delete immediately
      executeDelete(targetType, targetId);
    } else {
      // Otherwise, open modal
      setItemToDelete({ type: targetType, id: targetId });
      setDoNotShowAgain(false); // Reset checkbox
      setIsModalOpen(true);
    }
  };

  // 3. Confirm Delete Action (From Modal)
  const confirmUnlike = async () => {
    if (!itemToDelete) return;

    // Save preference if checkbox is checked
    if (doNotShowAgain) {
      localStorage.setItem('skip_unlike_confirmation', 'true');
    }

    // Execute Delete
    await executeDelete(itemToDelete.type, itemToDelete.id);

    // Close Modal
    setIsModalOpen(false);
    setItemToDelete(null);
  };

  const cancelUnlike = () => {
    setIsModalOpen(false);
    setItemToDelete(null);
  };

  const handleLogout = async () => {
    await logout();
    router.push('/');
  };

  // Filter data
  const movies = likes.filter(l => l.targetType === 'Movie');
  const actors = likes.filter(l => l.targetType === 'Actor');
  const directors = likes.filter(l => l.targetType === 'Director');
  const currentItems = activeTab === 'Movie' ? movies : activeTab === 'Actor' ? actors : directors;

  if (!isAuthenticated) return null;

  return (
      <div className="min-h-screen bg-black text-white relative">
        <Header />

        <main className="container mx-auto px-4 pt-24 pb-16 relative z-10">

          {/* Top Actions */}
          <div className="flex items-center justify-end gap-3 mb-8">
            <button
                onClick={() => {
                  sessionStorage.setItem('questionnaire_from_mypage', 'true');
                  router.push('/questionnaire');
                }}
                className="px-4 py-2 rounded-full border border-gray-700 bg-gray-900 text-sm hover:bg-gray-800 transition"
            >
              취향 설정 변경
            </button>
            <button
                onClick={toggleDarkMode}
                className="px-4 py-2 rounded-full border border-gray-700 bg-gray-900 text-sm hover:bg-gray-800 transition"
                aria-pressed={darkMode}
            >
              {darkMode ? '라이트 모드' : '다크 모드'}
            </button>
          </div>

          {/* Profile Header */}
          <div className="mb-12">
            <div className="flex items-center gap-6 mb-8">
              <div className="w-20 h-20 rounded-full bg-red-600 flex items-center justify-center">
                <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              </div>
              <div>
                <h1 className="text-3xl font-bold mb-1">
                  {user?.name ? `${user.name}님의 페이지` : '마이페이지'}
                </h1>
                <p className="text-gray-400 text-sm mb-2">좋아요한 영화, 감독, 배우를 관리하세요</p>
                {user?.email && <p className="text-xs text-gray-500">계정: {user.email}</p>}
              </div>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-12">
              <div
                  onClick={() => setActiveTab('Movie')}
                  className={`p-6 rounded-xl border cursor-pointer transition ${activeTab === 'Movie' ? 'bg-gray-800 border-red-600' : 'bg-gray-900 border-gray-800 hover:bg-gray-800'}`}
              >
                <div className="text-3xl font-bold mb-2 text-center">{movies.length}</div>
                <div className="text-gray-400 text-sm text-center">좋아요한 영화</div>
              </div>
              <div
                  onClick={() => setActiveTab('Actor')}
                  className={`p-6 rounded-xl border cursor-pointer transition ${activeTab === 'Actor' ? 'bg-gray-800 border-red-600' : 'bg-gray-900 border-gray-800 hover:bg-gray-800'}`}
              >
                <div className="text-3xl font-bold mb-2 text-center">{actors.length}</div>
                <div className="text-gray-400 text-sm text-center">좋아요한 배우</div>
              </div>
              <div
                  onClick={() => setActiveTab('Director')}
                  className={`p-6 rounded-xl border cursor-pointer transition ${activeTab === 'Director' ? 'bg-gray-800 border-red-600' : 'bg-gray-900 border-gray-800 hover:bg-gray-800'}`}
              >
                <div className="text-3xl font-bold mb-2 text-center">{directors.length}</div>
                <div className="text-gray-400 text-sm text-center">좋아요한 감독</div>
              </div>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex gap-8 border-b border-gray-800 mb-8">
            {['Movie', 'Actor', 'Director'].map((tab) => (
                <button
                    key={tab}
                    onClick={() => setActiveTab(tab as any)}
                    className={`pb-4 text-lg font-medium transition relative ${
                        activeTab === tab ? 'text-white' : 'text-gray-500 hover:text-gray-300'
                    }`}
                >
                  {tab === 'Movie' ? '영화' : tab === 'Actor' ? '배우' : '감독'}
                  {activeTab === tab && <div className="absolute bottom-0 left-0 w-full h-1 bg-red-600 rounded-t" />}
                </button>
            ))}
          </div>

          {/* Grid Content */}
          {loading ? (
              <div className="text-center py-20 text-gray-500">로딩 중...</div>
          ) : currentItems.length === 0 ? (
              <div className="text-center py-20 text-gray-500">
                아직 좋아요한 {activeTab === 'Movie' ? '영화가' : activeTab === 'Actor' ? '배우가' : '감독이'} 없습니다.
              </div>
          ) : (
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-6">
                {currentItems.map((item) => {
                  const isMovie = item.targetType === 'Movie';
                  const imageUrl = isMovie ? item.target.posterUrl : item.target.profileUrl;
                  const title = isMovie ? item.target.title : item.target.name;
                  const linkUrl = isMovie ? `/movie/${item.target._id}` : `/${item.targetType.toLowerCase()}/${item.target._id}`;

                  return (
                      <div key={item.id} className="group relative">
                        <Link href={linkUrl} className="block relative aspect-[2/3] bg-gray-800 rounded-lg overflow-hidden mb-3 hover:ring-2 hover:ring-red-600 transition-all">
                          {imageUrl ? (
                              <Image
                                  src={imageUrl}
                                  alt={title || ''}
                                  fill
                                  className="object-cover group-hover:scale-105 transition duration-300"
                                  sizes="(max-width: 768px) 50vw, 20vw"
                              />
                          ) : (
                              <div className="w-full h-full flex items-center justify-center text-gray-600 font-bold text-xl bg-gray-900">
                                {title?.charAt(0)}
                              </div>
                          )}
                          <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition" />
                        </Link>

                        <div className="flex justify-between items-start gap-2">
                          <Link href={linkUrl} className="flex-1 min-w-0">
                            <h3 className="font-medium truncate hover:text-red-500 transition text-sm md:text-base">
                              {title}
                            </h3>
                            {isMovie && item.target.year && (
                                <p className="text-xs text-gray-500">{item.target.year}</p>
                            )}
                          </Link>

                          <button
                              onClick={(e) => {
                                e.preventDefault();
                                initiateUnlike(item.targetType, item.target._id); // Check preference & Open Modal
                              }}
                              className="text-red-600 hover:text-red-400 transition p-1"
                              title="좋아요 취소"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6">
                              <path d="M11.645 20.91l-.007-.003-.022-.012a15.247 15.247 0 01-.383-.218 25.18 25.18 0 01-4.244-3.17C4.688 15.36 2.25 12.174 2.25 8.25 2.25 5.322 4.714 3 7.688 3A5.5 5.5 0 0112 5.052 5.5 5.5 0 0116.313 3c2.973 0 5.437 2.322 5.437 5.25 0 3.925-2.438 7.111-4.739 9.256a25.175 25.175 0 01-4.244 3.17 15.247 15.247 0 01-.383.219l-.022.012-.007.004-.003.001a.752.752 0 01-.704 0l-.003-.001z" />
                            </svg>
                          </button>
                        </div>
                      </div>
                  );
                })}
              </div>
          )}

          <div className="mt-16 text-center">
            <button onClick={handleLogout} className="px-6 py-3 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition text-sm">
              로그아웃
            </button>
          </div>
        </main>

        {/* --- Custom Modal UI --- */}
        {isModalOpen && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm px-4">
              <div className="bg-[#1a1d29] border border-gray-700 p-6 rounded-xl shadow-2xl w-full max-w-sm transform transition-all scale-100">

                {/* Header */}
                <h3 className="text-xl font-bold text-white mb-3 text-center">알림</h3>

                {/* Body */}
                <p className="text-gray-300 text-center mb-6 leading-relaxed">
                  정말 찜 목록에서 삭제하시겠습니까?
                </p>

                {/* Checkbox: Do not show again */}
                <div className="flex items-center justify-center mb-6 gap-2 cursor-pointer" onClick={() => setDoNotShowAgain(!doNotShowAgain)}>
                  <div className={`w-5 h-5 rounded border flex items-center justify-center transition ${doNotShowAgain ? 'bg-red-600 border-red-600' : 'border-gray-500 bg-transparent'}`}>
                    {doNotShowAgain && (
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 text-white" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                    )}
                  </div>
                  <span className="text-sm text-gray-400 select-none">다시 보지 않기</span>
                </div>

                {/* Buttons */}
                <div className="flex gap-3">
                  <button
                      onClick={cancelUnlike}
                      className="flex-1 py-3 px-4 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-lg font-medium transition"
                  >
                    취소
                  </button>
                  <button
                      onClick={confirmUnlike}
                      className="flex-1 py-3 px-4 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition"
                  >
                    삭제
                  </button>
                </div>

              </div>
            </div>
        )}

      </div>
  );
}
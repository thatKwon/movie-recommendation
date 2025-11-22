'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import { useAuth } from '@/context/AuthContext';
import { recommendationsAPI, userAPI, moviesAPI } from '@/lib/api';
import Header from '@/components/Header';
import Carousel from '@/components/Carousel';

// Define types locally
interface Movie {
  id: number;
  title: string;
  year: string;
  posterUrl?: string;
  rating?: number;
  userLiked?: boolean;
  genres?: string[];
}

interface Section {
  title: string;
  movies: Movie[];
  originalTitle?: string;
}

// Master list of genres in Korean to match DB
const ALL_GENRES = [
  '액션', '드라마', '코미디', '로맨스', '스릴러', '공포',
  'SF', '판타지', '애니메이션', '다큐멘터리', '범죄', '가족'
];

export default function HomePage() {
  const router = useRouter();
  const { user, isAuthenticated, loading: authLoading } = useAuth();

  // --- Search State ---
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState<Movie[]>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [loadingSearch, setLoadingSearch] = useState(false);
  const [searchError, setSearchError] = useState('');

  // --- Content State ---
  const [newMovies, setNewMovies] = useState<Movie[]>([]);
  const [userGenreSections, setUserGenreSections] = useState<Section[]>([]);
  const [otherGenreSections, setOtherGenreSections] = useState<Section[]>([]);

  // --- Loading Indicators ---
  const [loadingNew, setLoadingNew] = useState(false); // Default false to prevent flash
  const [loadingUserGenres, setLoadingUserGenres] = useState(false);
  const [loadingOtherGenres, setLoadingOtherGenres] = useState(false);

  // --- Main Data Fetching ---
  const loadData = useCallback(async () => {
    // 1. Wait for Auth to finish loading
    if (authLoading) return;

    // 2. FIX: If NOT authenticated, clear data and STOP here.
    if (!isAuthenticated) {
      setNewMovies([]);
      setUserGenreSections([]);
      setOtherGenreSections([]);
      return;
    }

    // 3. If Authenticated, start fetching
    setLoadingNew(true);
    moviesAPI.getNew(20)
        .then(res => setNewMovies(res.data.movies || []))
        .catch(err => console.error('New movies failed', err))
        .finally(() => setLoadingNew(false));

    // Logic to split "User Liked" vs "The Rest"
    const userPrefs = (user?.preferredGenres) ? user.preferredGenres : [];
    const restGenres = ALL_GENRES.filter(g => !userPrefs.includes(g));

    // --- Phase 1: Load User Preferred Genres ---
    setLoadingUserGenres(true);
    if (userPrefs.length > 0) {
      try {
        const response = await moviesAPI.getByGenres(userPrefs);
        const sections = response.data.sections || [];
        setUserGenreSections(sections);
      } catch (err) {
        console.error('User genres failed:', err);
      }
    } else {
      setUserGenreSections([]);
    }
    setLoadingUserGenres(false);

    // --- Phase 2: Load Remaining Genres (Sequential) ---
    if (restGenres.length > 0) {
      setLoadingOtherGenres(true);
      try {
        const response = await moviesAPI.getByGenres(restGenres);
        const sections = response.data.sections || [];
        setOtherGenreSections(sections);
      } catch (err) {
        console.error('Other genres failed:', err);
      } finally {
        setLoadingOtherGenres(false);
      }
    }
  }, [isAuthenticated, user, authLoading]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // --- Search Handler ---
  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isAuthenticated) {
      router.push('/login');
      return;
    }
    if (!query.trim()) return;

    setLoadingSearch(true);
    setSearchError('');
    setHasSearched(true);
    try {
      const profileRes = await userAPI.getProfile();
      const userPrefs = profileRes.data?.user || {};
      const response = await recommendationsAPI.search(query, {
        genres: userPrefs.preferredGenres,
        actors: userPrefs.preferredActors,
        years: userPrefs.preferredYears
      });
      setSearchResults(response.data.movies || []);
    } catch (err) {
      setSearchError('검색에 실패했습니다.');
    } finally {
      setLoadingSearch(false);
      setQuery('');
    }
  };

  // --- Display Name Logic ---
  const displayName = isAuthenticated && user?.name ? user.name : null;

  return (
      <div className="min-h-screen bg-black text-white">
        <Header />

        {/* Hero / Search Section */}
        <section id="search" className="relative min-h-[60vh] flex items-center justify-center">
          <div className="absolute inset-0 bg-gradient-to-b from-blue-900/20 to-black"></div>
          <div className="absolute inset-0" style={{ backgroundImage: 'radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px)', backgroundSize: '50px 50px' }}></div>
          <div className="container mx-auto px-4 py-32 relative z-10">
            <div className="max-w-4xl mx-auto text-center">

              {/* Title */}
              <div className="mb-8 animate-fade-in-up">
                <h1 className="text-4xl md:text-6xl font-bold mb-4 leading-tight">
                  {displayName ? (
                      <>
                        <span className="text-red-500">{displayName}</span>님을 위한 영화
                      </>
                  ) : (
                      "당신을 위한 영화"
                  )}
                </h1>

                {isAuthenticated ? (
                    <p className="text-xl md:text-2xl text-gray-300 font-light">
                      선호하는 취향과 오늘의 기분에 맞춰 엄선된 추천작을 만나보세요.
                    </p>
                ) : (
                    <p className="text-xl md:text-2xl text-gray-300 font-light">
                      오늘 당신의 마음을 움직일 특별한 이야기를 준비했습니다.
                    </p>
                )}
              </div>

              <form onSubmit={handleSearch} className="flex gap-4 mt-8">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="예: 우주를 배경으로 한 감동적인 영화"
                    className="flex-1 px-6 py-4 bg-gray-800/50 backdrop-blur border border-gray-700 rounded-lg focus:outline-none focus:border-red-500 text-lg"
                />
                <button
                    type="submit"
                    disabled={loadingSearch}
                    className="px-8 py-4 bg-red-600 text-white rounded-lg font-medium hover:bg-red-700 transition disabled:opacity-50"
                >
                  {loadingSearch ? '...' : '검색'}
                </button>
              </form>
              {searchError && <div className="mt-4 text-red-500">{searchError}</div>}
            </div>
          </div>
        </section>

        {/* Search Results */}
        {hasSearched && (
            <section className="container mx-auto px-4 py-8 relative z-10">
              <h2 className="text-2xl font-bold mb-6">검색 결과</h2>
              {loadingSearch ? <p>검색 중...</p> : (
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    {searchResults.map(movie => (
                        <div key={movie.id} onClick={() => router.push(`/movie/${movie.id}`)} className="cursor-pointer group">
                          <div className="relative aspect-[2/3] bg-gray-800 rounded overflow-hidden mb-2">
                            {movie.posterUrl && <Image src={movie.posterUrl} alt={movie.title} fill className="object-cover group-hover:scale-105 transition" />}
                          </div>
                          <h3 className="text-sm font-medium">{movie.title}</h3>
                        </div>
                    ))}
                  </div>
              )}
            </section>
        )}

        {/* Main Content: ONLY Show if Authenticated */}
        {isAuthenticated && (
            <div className="pb-16 relative z-10 container mx-auto px-4">
              {/* 1. New Movies */}
              <Carousel title="새로운 영화" movies={newMovies} loading={loadingNew} />

              {/* 2. User Liked Genres */}
              {userGenreSections.length > 0 && (
                  <div className="mt-12 mb-8">
                    <div className="flex items-center gap-2 mb-6 pl-2 border-l-4 border-red-600">
                      <h2 className="text-2xl font-bold text-red-500">회원님을 위한 추천 장르</h2>
                    </div>
                    {userGenreSections.map(section => (
                        <Carousel
                            key={`user-${section.title}`}
                            title={section.title}
                            movies={section.movies}
                            loading={loadingUserGenres}
                        />
                    ))}
                  </div>
              )}

              {/* 3. Other Genres */}
              {otherGenreSections.length > 0 && (
                  <div className="mt-16 mb-8">
                    <div className="flex items-center gap-2 mb-6 pl-2 border-l-4 border-red-600">
                      <h2 className="text-2xl font-bold text-red-500">다른 장르 탐험하기</h2>
                    </div>
                    {otherGenreSections.map(section => (
                        <Carousel
                            key={`other-${section.title}`}
                            title={section.title}
                            movies={section.movies}
                            loading={loadingOtherGenres}
                        />
                    ))}
                  </div>
              )}

              {/* Loader for Other Genres */}
              {loadingOtherGenres && !loadingUserGenres && (
                  <div className="opacity-50 mt-12">
                    <Carousel title="다른 장르 불러오는 중..." movies={[]} loading={true} />
                  </div>
              )}
            </div>
        )}
      </div>
  );
}

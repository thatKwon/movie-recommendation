'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { moviesAPI, actorsAPI, directorsAPI, likesAPI } from '@/lib/api';
import Header from '@/components/Header';
import MoviePosterOverlay from '@/components/MoviePosterOverlay';

export default function SearchPage() {
  const router = useRouter();
  const { isAuthenticated } = useAuth();
  const [activeTab, setActiveTab] = useState<'영화' | '배우' | '감독'>('영화');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any[]>([]);
  const [peopleResults, setPeopleResults] = useState<any[]>([]);
  const [actorAllMovies, setActorAllMovies] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [genre, setGenre] = useState('');
  const [year, setYear] = useState('');
  const [likedMap, setLikedMap] = useState<Record<string, boolean>>({});

  // Restore previous search state on mount (when navigating back)
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem('search_state_v1');
      if (!raw) return;
      const s = JSON.parse(raw);
      if (s) {
        setActiveTab(s.activeTab ?? '영화');
        setQuery(s.query ?? '');
        setGenre(s.genre ?? '');
        setYear(s.year ?? '');
        setResults(s.results ?? []);
        setPeopleResults(s.peopleResults ?? []);
      }
    } catch {}
  }, []);

  const saveState = (override?: Partial<{ results: any[]; peopleResults: any[] }>) => {
    try {
      const payload = {
        activeTab,
        query,
        genre,
        year,
        results: override?.results ?? results,
        peopleResults: override?.peopleResults ?? peopleResults
      };
      sessionStorage.setItem('search_state_v1', JSON.stringify(payload));
    } catch {}
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!isAuthenticated) {
      router.push('/login');
      return;
    }

    if (!query.trim()) {
      return;
    }

    setLoading(true);
    try {
      if (activeTab === '영화') {
        const response = await moviesAPI.search(query, {
          genre: genre || undefined,
          year: year || undefined
        });
        let movieList = response.data.movies || [];

        // Fallback: if no direct movie matches, try actor/director search and flatten filmography
        if ((!movieList || movieList.length === 0) && query.trim().length > 0) {
          try {
            const [actorsRes, directorsRes] = await Promise.all([
              actorsAPI.search(query),
              directorsAPI.search(query)
            ]);

            const movieMap: Record<string, any> = {};
            const collect = (arr: any[]) => {
              for (const p of arr || []) {
                for (const m of p.movies || []) {
                  if (!movieMap[m.id]) movieMap[m.id] = m;
                }
              }
            };
            collect(actorsRes.data?.actors || []);
            collect(directorsRes.data?.directors || []);
            movieList = Object.values(movieMap);
          } catch {}
        }

        // Sort by popularity (likeCount + viewCount) descending
        movieList.sort((a: any, b: any) => {
          const popularityA = (a.likeCount || 0) + (a.viewCount || 0);
          const popularityB = (b.likeCount || 0) + (b.viewCount || 0);
          return popularityB - popularityA;
        });

        // Check like status for all movies
        if (isAuthenticated && movieList.length > 0) {
          try {
            const items = movieList.map((m: any) => ({ type: 'Movie', id: String(m.id) }));
            const likeRes = await likesAPI.check(items);
            const liked = likeRes.data?.liked || {};
            const map: Record<string, boolean> = {};
            for (const item of items) {
              map[item.id] = Boolean(liked[`Movie_${item.id}`]);
            }
            setLikedMap(map);
          } catch (err) {
            // Ignore errors
          }
        }

        setResults(movieList);
        setPeopleResults([]);
        saveState({ results: movieList, peopleResults: [] });
      } else if (activeTab === '배우') {
        const response = await actorsAPI.search(query);
        setPeopleResults(response.data.actors || []);
        setResults([]);
        setActorAllMovies([]);
        saveState({ results: [], peopleResults: response.data.actors || [] });
      } else if (activeTab === '감독') {
        const response = await directorsAPI.search(query);
        setPeopleResults(response.data.directors || []);
        setResults([]);
        saveState({ results: [], peopleResults: response.data.directors || [] });
      }
    } catch (err) {
      // swallow; show empty
      setResults([]);
      setPeopleResults([]);
      setActorAllMovies([]);
      saveState({ results: [], peopleResults: [] });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white">
      <Header />

      <div className="container mx-auto px-4 pt-24 pb-16">
        <h1 className="text-4xl font-bold mb-8">검색</h1>

        {/* Search Input */}
        <div className="mb-8">
          <div className="relative">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch(e)}
              placeholder="영화 제목, 감독, 배우를 검색하세요"
              className="w-full px-6 py-4 bg-gray-900 border border-gray-700 rounded-lg focus:outline-none focus:border-red-500 text-lg pl-12"
            />
            <svg
              className="w-6 h-6 text-gray-400 absolute left-4 top-1/2 transform -translate-y-1/2"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
          </div>

          {/* Simple filters (genre/year) */}
          {activeTab === '영화' && (
            <div className="mt-4 flex gap-3">
              <select
                value={genre}
                onChange={(e) => setGenre(e.target.value)}
                className="px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-sm"
              >
                <option value="">장르 전체</option>
                <option value="Action">액션</option>
                <option value="Drama">드라마</option>
                <option value="Comedy">코미디</option>
                <option value="Romance">로맨스</option>
                <option value="Thriller">스릴러</option>
                <option value="Horror">공포</option>
                <option value="Science Fiction">SF</option>
                <option value="Fantasy">판타지</option>
                <option value="Animation">애니메이션</option>
                <option value="Documentary">다큐멘터리</option>
                <option value="Crime">범죄</option>
                <option value="Family">가족</option>
              </select>
              <input
                value={year}
                onChange={(e) => setYear(e.target.value)}
                type="number"
                placeholder="년도"
                className="w-28 px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-sm"
              />
              <button
                onClick={handleSearch}
                className="px-4 py-2 bg-red-600 text-white rounded-lg text-sm hover:bg-red-700"
              >
                검색
              </button>
            </div>
          )}
        </div>

        {/* Tabs */}
        <div className="flex gap-4 mb-8 border-b border-gray-800">
          {['영화', '배우', '감독'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab as any)}
              className={`px-4 py-2 transition ${
                activeTab === tab
                  ? 'text-white border-b-2 border-red-500'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Results */}
        {!isAuthenticated ? (
          <div className="text-center py-16">
            <p className="text-gray-400 mb-4">검색하려면 로그인이 필요합니다</p>
            <button
              onClick={() => router.push('/login')}
              className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition"
            >
              로그인하러 가기
            </button>
          </div>
        ) : results.length === 0 && peopleResults.length === 0 && !loading && !query ? (
          <div className="text-center py-16">
            <p className="text-gray-400">검색어를 입력해주세요</p>
          </div>
        ) : results.length === 0 && peopleResults.length === 0 && query && !loading ? (
          <div className="text-center py-16">
            <p className="text-gray-400">검색 결과가 없습니다</p>
          </div>
        ) : loading ? (
          <div className="text-center py-16">
            <p className="text-gray-400">검색 중...</p>
          </div>
        ) : activeTab === '영화' ? (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-6">
            {results.map((item) => (
              <div
                key={item.id}
                onClick={() => {
                  saveState();
                  router.push(`/movie/${item.id}`);
                }}
                className="cursor-pointer group"
              >
                <div className="relative aspect-[2/3] bg-gray-800 rounded-lg overflow-hidden mb-3 group-hover:ring-2 group-hover:ring-red-500 transition">
                  {item.posterUrl ? (
                    <img
                      src={item.posterUrl}
                      alt={item.title}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-500">
                      No Image
                    </div>
                  )}
                  {/* Rating and Like Button Overlay */}
                  <MoviePosterOverlay
                    movieId={item.id}
                    rating={item.rating}
                    initiallyLiked={likedMap[item.id] || false}
                    onLikeChange={(liked) => {
                      setLikedMap(prev => ({ ...prev, [item.id]: liked }));
                    }}
                  />
                </div>
                <h3 className="font-medium line-clamp-2">{item.title}</h3>
                <p className="text-sm text-gray-400">{item.year}</p>
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-8">
            {peopleResults.map((p) => (
              <div key={p.id} className="bg-gray-900 rounded-lg p-5 border border-gray-800">
                <div className="flex items-center gap-4 mb-4 cursor-pointer" onClick={() => (activeTab === '배우' ? router.push(`/actor/${p.id}`) : undefined)}>
                  <div className="w-16 h-16 rounded-full bg-gray-800 overflow-hidden flex items-center justify-center">
                    {p.profileUrl ? (
                      <img src={p.profileUrl} alt={p.name} className="w-full h-full object-cover" />
                    ) : (
                      <span className="text-gray-500 text-xs">No Image</span>
                    )}
                  </div>
                  <div>
                    <div className="text-lg font-semibold">{p.name}</div>
                    {p.nameEnglish && <div className="text-gray-400 text-sm">{p.nameEnglish}</div>}
                  </div>
                </div>

                {/* If a single actor matches, show all of their movies in a grid */}
                {activeTab === '배우' && peopleResults.length === 1 && p.id === peopleResults[0].id ? (
                  <ActorFullFilmography actorId={p.id} cached={actorAllMovies} onLoad={setActorAllMovies} />
                ) : (
                  Array.isArray(p.movies) && p.movies.length > 0 ? (
                    <div className="flex gap-4 overflow-x-auto pb-1">
                      {p.movies.map((m: any) => (
                        <div key={m.id} className="w-[120px] flex-shrink-0 cursor-pointer" onClick={() => { saveState(); router.push(`/movie/${m.id}`); }}>
                          <div className="aspect-[2/3] bg-gray-800 rounded-lg overflow-hidden mb-2">
                            {m.posterUrl ? (
                              <img src={m.posterUrl} alt={m.title} className="w-full h-full object-cover" />
                            ) : (
                              <div className="w-full h-full flex items-center justify-center text-gray-500 text-xs">No Image</div>
                            )}
                          </div>
                          <div className="text-xs line-clamp-2">{m.title}</div>
                          <div className="text-[10px] text-gray-400">{m.year}</div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-gray-400 text-sm">관련 영화가 없습니다</div>
                  )
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// Inline client helper component to fetch full actor filmography
function ActorFullFilmography({ actorId, cached, onLoad }: { actorId: string; cached: any[]; onLoad: (m: any[]) => void; }) {
  // this is a client component body embedded at the end of the page file
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const router = useRouter();
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const { isAuthenticated } = useAuth();
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const [loading, setLoading] = useState(false);
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const [likedMap, setLikedMap] = useState<Record<string, boolean>>({});

  // eslint-disable-next-line react-hooks/rules-of-hooks
  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      if (cached && cached.length > 0) return;
      setLoading(true);
      try {
        const res = await actorsAPI.getById(actorId);
        const movies = res.data?.actor?.movies || [];
        if (!cancelled) onLoad(movies);
      } catch {
        if (!cancelled) onLoad([]);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    run();
    return () => {
      cancelled = true;
    };
  }, [actorId]);

  // Check like status for movies
  // eslint-disable-next-line react-hooks/rules-of-hooks
  useEffect(() => {
    if (!isAuthenticated || !cached || cached.length === 0) return;
    const items = cached.map((m: any) => ({ type: 'Movie', id: String(m.id) }));
    likesAPI
      .check(items)
      .then((res) => {
        const map: Record<string, boolean> = {};
        const liked = res.data?.liked || {};
        for (const it of items) {
          map[it.id] = Boolean(liked[`Movie_${it.id}`]);
        }
        setLikedMap(map);
      })
      .catch(() => {});
  }, [isAuthenticated, cached]);

  if (loading && (!cached || cached.length === 0)) {
    return (
      <div className="text-center py-6 text-gray-400">전체 출연작을 불러오는 중...</div>
    );
  }

  if (!cached || cached.length === 0) {
    return <div className="text-gray-400 text-sm">관련 영화가 없습니다</div>;
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-6">
      {cached.map((m: any) => (
        <div key={m.id} className="cursor-pointer" onClick={() => router.push(`/movie/${m.id}`)}>
          <div className="relative aspect-[2/3] bg-gray-800 rounded-lg overflow-hidden mb-2">
            {m.posterUrl ? (
              <img src={m.posterUrl} alt={m.title} className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-gray-500 text-xs">No Image</div>
            )}
            {/* Rating and Like Button Overlay */}
            <MoviePosterOverlay
              movieId={String(m.id)}
              rating={m.rating}
              initiallyLiked={likedMap[String(m.id)] || false}
              onLikeChange={(liked) => {
                setLikedMap(prev => ({ ...prev, [String(m.id)]: liked }));
              }}
            />
          </div>
          <div className="text-sm font-medium line-clamp-2">{m.title}</div>
          <div className="text-xs text-gray-400">{m.year}</div>
        </div>
      ))}
    </div>
  );
}

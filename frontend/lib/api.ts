import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5001';

export const api = axios.create({
    baseURL: API_BASE,
    withCredentials: true
});

// ---------------- Access Token Handling ----------------
let accessToken: string | null = null;

export const setAccessToken = (token: string | null) => {
    accessToken = token;
    if (token) {
        api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    } else {
        delete api.defaults.headers.common['Authorization'];
    }
};

// ================== USER API ==================
export const userAPI = {
    getStats: () => api.get('/api/user/stats'),
    getLikedMovies: () => api.get('/api/user/liked-movies'),
    getLikedActors: () => api.get('/api/user/liked-actors'),
    getLikedDirectors: () => api.get('/api/user/liked-directors'),

    getSearchHistory: (limit = 10) =>
        api.get('/api/user/search-history', { params: { limit } }),

    submitQuestionnaire: (data: any) =>
        api.post('/api/user/questionnaire', data),

    getProfile: () => api.get('/api/user/profile'),
    updateProfile: (data: any) => api.patch('/api/user/profile', data),
};

// ================== MOVIES API ==================
export const moviesAPI = {
    getMovie: async (id: string) => {
        const { data } = await api.get(`/api/movies/${id}`);
        return data.movie;
    },

    getById: (id: string, tmdbId?: number) =>
        api.get(`/api/movies/${id}`, tmdbId ? { params: { tmdbId } } : {}),

    search: (query: string, filters?: any) =>
        api.get('/api/movies/search', { params: { q: query, ...filters } }),

    getByGenres: (genres: string[]) =>
        api.get('/api/movies/by-genres', {
            params: { genres: genres.join(',') }
        }),

    getNew: (limit?: number) =>
        api.get('/api/movies/new', { params: limit ? { limit } : {} }),

    getPopular: (limit?: number) =>
        api.get('/api/movies/popular', { params: limit ? { limit } : {} }),

    trackClick: (id: string) => api.post(`/api/movies/${id}/click`)
};

// ================== LIKES API ==================
export const likesAPI = {
    // ADDED: Fetch all liked items (Movies, Actors, Directors)
    getAll: () => api.get('/api/likes'),

    create: (targetType: string, targetId: string) =>
        api.post('/api/likes', { targetType, targetId }),

    delete: (likeId: string) => api.delete(`/api/likes/${likeId}`),

    deleteByTarget: (targetType: string, targetId: string) =>
        api.delete('/api/likes', { params: { targetType, targetId } }),

    check: (items: { type: string; id: string }[]) =>
        api.get('/api/likes/check', {
            params: { items: JSON.stringify(items) }
        }),

    toggle: (targetType: string, targetId: string) =>
        api.post('/api/likes/toggle', { targetType, targetId })
};

// ================== ACTORS API ==================
export const actorsAPI = {
    getById: (id: string) => api.get(`/api/actors/${id}`),
    search: (q: string) => api.get('/api/actors/search', { params: { q } })
};

// ================== DIRECTORS API ==================
export const directorsAPI = {
    getById: async (id: string) => {
        const { data } = await api.get(`/api/directors/${id}`);
        return data.director;
    },
    search: (q: string) => api.get('/api/directors/search', { params: { q } })
};

// ================== RECOMMENDATIONS API ==================
export const recommendationsAPI = {
    search: (query: string, userPreferences?: any) =>
        api.post('/api/recommendations/search', {
            query,
            userPreferences
        })
};

// ================== AUTH API ==================
export const authAPI = {
    login: (email: string, password: string) =>
        api.post('/api/auth/login', { email, password }),

    signup: (data: { email: string; password: string; name: string }) =>
        api.post('/api/auth/signup', data),

    logout: () =>
        api.post('/api/auth/logout'),

    refresh: () =>
        api.post('/api/auth/refresh')
};
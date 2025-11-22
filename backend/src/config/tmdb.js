/**
 * TMDB API configuration
 * The Movie Database (TMDB) API settings
 */

const TMDB_CONFIG = {
  baseUrl: process.env.TMDB_BASE_URL || 'https://api.themoviedb.org/3',
  apiKey: process.env.TMDB_API_KEY,
  imageBaseUrl: 'https://image.tmdb.org/t/p',

  // Image sizes
  posterSizes: {
    small: 'w185',
    medium: 'w342',
    large: 'w500',
    original: 'original'
  },
  backdropSizes: {
    small: 'w300',
    medium: 'w780',
    large: 'w1280',
    original: 'original'
  },
  profileSizes: {
    small: 'w45',
    medium: 'w185',
    large: 'h632',
    original: 'original'
  },

  // Language settings
  language: {
    korean: 'ko-KR',
    english: 'en-US'
  }
};

/**
 * Get full image URL from TMDB path
 * @param {string} path - Image path from TMDB API
 * @param {string} size - Image size (e.g., 'w500')
 * @returns {string} Full image URL
 */
const getImageUrl = (path, size = 'w500') => {
  if (!path) return null;
  return `${TMDB_CONFIG.imageBaseUrl}/${size}${path}`;
};

module.exports = {
  TMDB_CONFIG,
  getImageUrl
};

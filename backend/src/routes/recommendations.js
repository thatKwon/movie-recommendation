// javascript
const express = require('express');
const router = express.Router();
const Search = require('../models/Search');
const Like = require('../models/Like');
const { authenticateToken, optionalAuth } = require('../middleware/auth');
const {
  searchWithRecommendation,
  getHomeRecommendations
} = require('../services/recommendationService');
const { sanitizePreferences } = require('../utils/sanitizePreferences');
const { logError } = require('../utils/errorHandler');

/**
 * POST /api/recommendations/search
 * Natural language movie search with LLM-powered recommendations
 * Auth: Required
 */
router.post('/search', authenticateToken, async (req, res) => {
  try {
    const { query } = req.body;
    const rawPrefs = req.body?.userPreferences || {};
    const cleanedPrefs = sanitizePreferences(rawPrefs);

    // Validation
    if (!query) {
      return res.status(400).json({ error: 'Query required' });
    }

    if (query.length > 500) {
      return res.status(400).json({ error: 'Query too long (max 500 characters)' });
    }

    // Call recommendation service (pass cleaned preferences)
    let movies;
    try {
      movies = await searchWithRecommendation(req.user.userId, query, cleanedPrefs);
    } catch (error) {
      return res.status(503).json({ error: 'Recommendation service unavailable' });
    }

    // Save search history
    const search = await Search.create({
      userId: req.user.userId,
      query,
      resultMovieIds: (movies || []).map(m => m._id)
    });

    // Format response
    const movieResults = (movies || []).map(movie => ({
      id: movie._id,
      title: movie.title,
      titleEnglish: movie.titleEnglish,
      year: movie.year,
      genres: movie.genres,
      posterUrl: movie.posterUrl,
      rating: movie.rating,
      relevanceScore: movie.relevanceScore || 0
    }));

    res.status(200).json({
      movies: movieResults,
      searchId: search._id
    });
  } catch (error) {
    logError(error, req);
    res.status(500).json({ error: 'Server error' });
  }
});

/**
 * GET /api/recommendations/home
 * Get personalized recommendations for home page
 * Auth: Optional (shows default genres if not authenticated)
 */
router.get('/home', optionalAuth, async (req, res) => {
  try {
    // Accept optional preferences from body or query (query may be JSON string)
    let rawPrefs = {};
    if (req.body && req.body.userPreferences) {
      rawPrefs = req.body.userPreferences;
    } else if (req.query && req.query.userPreferences) {
      try {
        rawPrefs = JSON.parse(req.query.userPreferences);
      } catch (e) {
        rawPrefs = {};
      }
    }
    const cleanedPrefs = sanitizePreferences(rawPrefs);

    // Call recommendation service with sanitized preferences
    // If not authenticated, userId will be null and service will use defaults
    const userId = req.user ? req.user.userId : null;
    let sections;
    try {
      sections = await getHomeRecommendations({ userId, userPreferences: cleanedPrefs });
    } catch (error) {
      // Only return error if algorithm is configured but fails to respond
      // If not configured, service returns empty sections (no error)
      const errorMessage = error.message || 'Failed to get recommendations';
      console.error('Home recommendations error:', errorMessage);
      
      // Return 503 Service Unavailable only if algorithm endpoint is configured but failed
      if (process.env.RECOMMENDATION_API_URL) {
        return res.status(503).json({
          error: '추천 서비스를 사용할 수 없습니다. 잠시 후 다시 시도해주세요.',
          errorCode: 'SERVICE_UNAVAILABLE'
        });
      }
      // If not configured, return empty sections (no error)
      sections = [];
    }

    // Format response with userLiked status if authenticated
    const formattedSections = await Promise.all((sections || []).map(async (section) => {
      const moviesWithLiked = await Promise.all((section.movies || []).map(async (movie) => {
        let userLiked = false;
        if (req.user) {
          const like = await Like.findOne({
            userId: req.user.userId,
            targetType: 'movie',
            targetId: movie._id
          });
          userLiked = !!like;
        }
        return {
          id: movie._id,
          title: movie.title,
          titleEnglish: movie.titleEnglish,
          year: movie.year,
          genres: movie.genres,
          posterUrl: movie.posterUrl,
          rating: movie.rating,
          likeCount: movie.likeCount,
          userLiked
        };
      }));
      return {
        title: section.title,
        movies: moviesWithLiked
      };
    }));

    res.status(200).json({ sections: formattedSections });
  } catch (error) {
    logError(error, req);
    // Soft-fail to empty sections on unexpected error
    res.status(200).json({ sections: [] });
  }
});

module.exports = router;

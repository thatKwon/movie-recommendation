const express = require('express');
const router = express.Router();
const User = require('../models/User');
const Like = require('../models/Like');
const Search = require('../models/Search');
const Movie = require('../models/Movie');
const Actor = require('../models/Actor');
const Director = require('../models/Director'); // Explicitly require Director
const { authenticateToken } = require('../middleware/auth');

// All user routes require authentication
router.use(authenticateToken);

/**
 * GET /api/user/profile
 * Get current user's full profile
 */
router.get('/profile', async (req, res) => {
  try {
    const user = await User.findById(req.user.userId).select('-password -refreshToken');

    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    res.status(200).json({
      user: {
        id: user._id,
        email: user.email,
        // --- FIX: Return name for UI ---
        name: user.name,
        // ------------------------------
        darkMode: user.darkMode,
        preferredGenres: user.preferredGenres,
        preferredActors: user.preferredActors,
        preferredDirectors: user.preferredDirectors,
        preferredYears: user.preferredYears,
        createdAt: user.createdAt
      }
    });
  } catch (error) {
    console.error('Get profile error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

/**
 * PATCH /api/user/profile
 * Update user profile settings
 */
router.patch('/profile', async (req, res) => {
  try {
    const { email, name, darkMode, preferredGenres, preferredActors, preferredDirectors, preferredYears } = req.body;

    const user = await User.findById(req.user.userId);

    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    // Update email if provided
    if (email && email !== user.email) {
      // Validate email format
      if (!email.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) {
        return res.status(400).json({ error: 'Invalid email format' });
      }

      // Check if email already in use
      const existingUser = await User.findOne({ email: email.toLowerCase() });
      if (existingUser && existingUser._id.toString() !== user._id.toString()) {
        return res.status(400).json({ error: 'Email already in use' });
      }

      user.email = email.toLowerCase();
    }

    // --- FIX: Update Name ---
    if (name) {
      user.name = name;
    }
    // ------------------------

    // Update other fields if provided
    if (darkMode !== undefined) {
      user.darkMode = darkMode;
    }

    if (preferredGenres !== undefined) {
      user.preferredGenres = preferredGenres;
    }

    if (preferredActors !== undefined) {
      user.preferredActors = preferredActors;
    }

    if (preferredDirectors !== undefined) {
      user.preferredDirectors = preferredDirectors;
    }

    if (preferredYears !== undefined) {
      user.preferredYears = preferredYears;
    }

    await user.save();

    res.status(200).json({
      user: {
        id: user._id,
        email: user.email,
        name: user.name, // Return updated name
        darkMode: user.darkMode,
        preferredGenres: user.preferredGenres,
        preferredActors: user.preferredActors,
        preferredDirectors: user.preferredDirectors,
        preferredYears: user.preferredYears
      }
    });
  } catch (error) {
    console.error('Update profile error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

/**
 * GET /api/user/stats
 * Get user statistics for profile page
 */
router.get('/stats', async (req, res) => {
  try {
    const userId = req.user.userId;

    // --- FIX: Use Capitalized Types to match new Logic ---
    const likedMoviesCount = await Like.countDocuments({
      userId,
      targetType: 'Movie'
    });

    const likedActorsCount = await Like.countDocuments({
      userId,
      targetType: 'Actor'
    });

    const likedDirectorsCount = await Like.countDocuments({
      userId,
      targetType: 'Director'
    });
    // ----------------------------------------------------

    // Count searches
    const searchesCount = await Search.countDocuments({ userId });

    res.status(200).json({
      stats: {
        likedMoviesCount,
        searchesCount,
        likedActorsCount,
        likedDirectorsCount
      }
    });
  } catch (error) {
    console.error('Get stats error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

/**
 * GET /api/user/liked-movies
 * Get all movies user has liked with pagination
 */
router.get('/liked-movies', async (req, res) => {
  try {
    const userId = req.user.userId;
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 20;
    const skip = (page - 1) * limit;

    // Get total count
    const total = await Like.countDocuments({
      userId,
      targetType: 'Movie' // Capitalized
    });

    // Get liked movies
    const likes = await Like.find({
      userId,
      targetType: 'Movie' // Capitalized
    })
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(limit);

    // Get movie details
    const movieIds = likes.map(like => like.targetId);
    const movies = await Movie.find({ _id: { $in: movieIds } });

    // Combine with liked dates
    const moviesWithLikedDate = movies.map(movie => {
      const like = likes.find(l => l.targetId.toString() === movie._id.toString());
      return {
        id: movie._id,
        tmdbId: movie.tmdbId,
        title: movie.title,
        titleEnglish: movie.titleEnglish,
        year: movie.year,
        genres: movie.genres,
        posterUrl: movie.posterUrl,
        rating: movie.rating,
        likedAt: like ? like.createdAt : null
      };
    });

    res.status(200).json({
      movies: moviesWithLikedDate,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit)
      }
    });
  } catch (error) {
    console.error('Get liked movies error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

/**
 * GET /api/user/liked-actors
 * Get all actors user has liked
 */
router.get('/liked-actors', async (req, res) => {
  try {
    const userId = req.user.userId;

    // Get liked actors
    const likes = await Like.find({
      userId,
      targetType: 'Actor' // Capitalized
    }).sort({ createdAt: -1 });

    // Get actor details
    const actorIds = likes.map(like => like.targetId);
    const actors = await Actor.find({ _id: { $in: actorIds } });

    // Combine with liked dates
    const actorsWithDetails = actors.map(actor => {
      const like = likes.find(l => l.targetId.toString() === actor._id.toString());
      return {
        id: actor._id,
        name: actor.name,
        nameEnglish: actor.nameEnglish,
        profileUrl: actor.profileUrl,
        movieCount: actor.movieIds ? actor.movieIds.length : 0,
        likedAt: like ? like.createdAt : null
      };
    });

    res.status(200).json({ actors: actorsWithDetails });
  } catch (error) {
    console.error('Get liked actors error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

/**
 * GET /api/user/liked-directors
 * Get all directors user has liked
 */
router.get('/liked-directors', async (req, res) => {
  try {
    const userId = req.user.userId;

    // Find likes for directors
    const likes = await Like.find({
      userId,
      targetType: 'Director' // Capitalized
    }).sort({ createdAt: -1 });

    const directorIds = likes.map(like => like.targetId);
    const directors = await Director.find({ _id: { $in: directorIds } });

    const directorsWithDetails = directors.map(director => {
      const like = likes.find(l => l.targetId.toString() === director._id.toString());
      return {
        id: director._id,
        name: director.name,
        nameEnglish: director.nameEnglish,
        profileUrl: director.profileUrl,
        movieCount: director.movieIds ? director.movieIds.length : 0,
        likedAt: like ? like.createdAt : null
      };
    });

    res.status(200).json({ directors: directorsWithDetails });
  } catch (error) {
    console.error('Get liked directors error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

/**
 * GET /api/user/search-history
 * Get user's search history
 */
router.get('/search-history', async (req, res) => {
  try {
    const userId = req.user.userId;
    const limit = Math.min(parseInt(req.query.limit) || 10, 50);

    const searches = await Search.find({ userId })
        .sort({ searchedAt: -1 })
        .limit(limit);

    const searchHistory = searches.map(search => ({
      id: search._id,
      query: search.query,
      searchedAt: search.searchedAt,
      resultCount: search.resultMovieIds.length
    }));

    res.status(200).json({ searches: searchHistory });
  } catch (error) {
    console.error('Get search history error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

/**
 * POST /api/user/questionnaire
 * Submit initial questionnaire answers
 */
router.post('/questionnaire', async (req, res) => {
  try {
    const { preferredGenres, preferredActors, preferredDirectors, preferredYears } = req.body;

    // Validation
    if (!preferredGenres || preferredGenres.length === 0) {
      return res.status(400).json({ error: 'At least one genre required' });
    }

    const user = await User.findById(req.user.userId);

    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    // Map Korean genre labels to English tags used in Movie.genres
    const genreMap = {
      '액션': 'Action',
      '드라마': 'Drama',
      '코미디': 'Comedy',
      '로맨스': 'Romance',
      '스릴러': 'Thriller',
      '공포': 'Horror',
      'SF': 'Science Fiction',
      '판타지': 'Fantasy',
      '애니메이션': 'Animation',
      '다큐멘터리': 'Documentary',
      '범죄': 'Crime',
      '가족': 'Family'
    };

    const normalizedGenres = Array.isArray(preferredGenres)
        ? preferredGenres.map(g => genreMap[g] || g).filter(Boolean)
        : [];

    // Update preferences (store English tags)
    user.preferredGenres = normalizedGenres;
    user.preferredActors = preferredActors || [];
    user.preferredDirectors = preferredDirectors || [];
    user.preferredYears = preferredYears || { min: 1990, max: 2024 };

    await user.save();

    res.status(200).json({
      message: 'Preferences saved',
      user: {
        id: user._id,
        name: user.name, // Return name here too
        preferredGenres: user.preferredGenres,
        preferredActors: user.preferredActors,
        preferredDirectors: user.preferredDirectors,
        preferredYears: user.preferredYears
      }
    });
  } catch (error) {
    console.error('Questionnaire error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

module.exports = router;
const mongoose = require('mongoose');

const movieSchema = new mongoose.Schema({
  tmdbId: {
    type: Number,
    required: true,
    unique: true
  },
  title: {
    type: String,
    required: true
  },
  titleEnglish: {
    type: String
  },
  year: {
    type: Number
  },
  genres: {
    type: [String],
    default: []
  },
  genresEnglish: {
    type: [String],
    default: []
  },
  plot: {
    type: String
  },
  plotEnglish: {
    type: String
  },
  posterUrl: {
    type: String
  },
  backdropUrl: {
    type: String
  },
  runtime: {
    type: Number
  },
  rating: {
    type: String // This is for Age Rating (e.g., "15세 관람가")
  },
  // --- FIX 1: Added voteAverage for Real Ratings (e.g. 7.3) ---
  voteAverage: {
    type: Number,
    default: 0.0
  },
  // -----------------------------------------------------------
  cast: [{
    actorId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Actor'
    },
    actorName: String,
    character: String,
    // --- FIX 2: Added profileUrl for Actor Images ---
    profileUrl: String,
    // -----------------------------------------------
    order: Number
  }],
  directors: [{
    directorId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Director'
    },
    directorName: String,
    // --- FIX 3: Added profileUrl for Director Images ---
    profileUrl: String
    // --------------------------------------------------
  }],
  likeCount: {
    type: Number,
    default: 0
  },
  viewCount: {
    type: Number,
    default: 0
  },
  platforms: {
    type: [String],
    default: []
  },
  tmdbLastFetched: {
    type: Date,
    default: Date.now
  }
}, {
  timestamps: true
});

// Create indexes for faster search and sorting
movieSchema.index({ year: 1 });
movieSchema.index({ genres: 1 });
movieSchema.index({ title: 'text', titleEnglish: 'text' });

module.exports = mongoose.model('Movie', movieSchema);
const mongoose = require('mongoose');

const directorSchema = new mongoose.Schema({
  tmdbId: {
    type: Number,
    required: true,
    unique: true
  },
  name: {
    type: String,
    required: true
  },
  nameEnglish: {
    type: String
  },
  profileUrl: {
    type: String
  },
  likeCount: {
    type: Number,
    default: 0
  },
  movieIds: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Movie'
  }]
}, {
  timestamps: true
});

// Create indexes
// directorSchema.index({ tmdbId: 1 }, { unique: true });
directorSchema.index({ name: 'text', nameEnglish: 'text' });

module.exports = mongoose.model('Director', directorSchema);

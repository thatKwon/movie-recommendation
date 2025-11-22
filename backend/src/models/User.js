const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  email: {
    type: String,
    required: [true, 'Email is required'],
    unique: true,
    lowercase: true,
    trim: true,
    match: [/^[^\s@]+@[^\s@]+\.[^\s@]+$/, 'Please provide a valid email address']
  },
  password: {
    type: String,
    required: [true, 'Password is required'],
    minlength: [6, 'Password must be at least 6 characters']
  },
  // ADDED: Nickname field
  name: {
    type: String,
    required: [true, 'Nickname is required'],
    trim: true,
    maxlength: [20, 'Nickname cannot be more than 20 characters']
  },
  preferredGenres: {
    type: [String],
    default: []
  },
  preferredActors: {
    type: [String],
    default: []
  },
  preferredDirectors: {
    type: [String],
    default: []
  },
  preferredYears: {
    type: {
      min: {
        type: Number,
        default: 1990
      },
      max: {
        type: Number,
        default: 2024
      }
    },
    default: { min: 1990, max: 2024 }
  },
  darkMode: {
    type: Boolean,
    default: true
  },
  refreshToken: {
    type: String,
    default: null
  }
}, {
  timestamps: true
});

userSchema.index({ refreshToken: 1 });

module.exports = mongoose.model('User', userSchema);
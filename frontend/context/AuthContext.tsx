'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';
import { setAccessToken as setApiAccessToken, userAPI } from '@/lib/api';

interface User {
  id: string;
  email: string;
  // --- FIX: Added name field ---
  name: string;
  // ---------------------------
  darkMode: boolean;
  preferredGenres?: string[];
  preferredActors?: string[];
  preferredDirectors?: string[];
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (token: string, userData: User) => void;
  logout: () => Promise<void>; // Updated signature to Promise
  updateUser: (userData: Partial<User>) => Promise<void>;
  loading: boolean; // Added loading state exposure
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const isAuthenticated = !!accessToken;

  const logout = useCallback(async () => {
    try {
      await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/auth/logout`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Authorization': `Bearer ${accessToken}` }
      });
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setAccessToken(null);
      setApiAccessToken(null);
      setUser(null);
      window.location.href = '/'; // Force redirect on logout
    }
  }, [accessToken]);

  // 1. Check for existing session (Refresh Token) on load
  useEffect(() => {
    const refreshToken = async () => {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/auth/refresh`, {
          method: 'POST',
          credentials: 'include'
        });

        if (response.ok) {
          const data = await response.json();
          setAccessToken(data.accessToken);
          setApiAccessToken(data.accessToken);
        } else {
          setIsLoading(false); // Stop loading if refresh fails (not logged in)
        }
      } catch (error) {
        console.error('Initial refresh failed:', error);
        setIsLoading(false);
      }
    };
    refreshToken();
  }, []);

  // 2. Fetch User Profile once we have an Access Token
  useEffect(() => {
    const fetchProfile = async () => {
      if (accessToken) {
        try {
          const response = await userAPI.getProfile();
          setUser(response.data.user);
          console.log('Profile loaded:', response.data.user);
        } catch (error) {
          console.error('Failed to fetch profile:', error);
          // If profile fetch fails (e.g. invalid token), log out
          setAccessToken(null);
          setApiAccessToken(null);
        } finally {
          setIsLoading(false);
        }
      }
    };

    if (accessToken) {
      fetchProfile();
    }
  }, [accessToken]);

  const login = (token: string, userData: User) => {
    setAccessToken(token);
    setApiAccessToken(token);
    setUser(userData);
  };

  const updateUser = async (userData: Partial<User>) => {
    if (!user) return;
    const updatedUser = { ...user, ...userData };
    setUser(updatedUser);
    try {
      const response = await userAPI.updateProfile(userData);
      setUser(response.data.user);
    } catch (error) {
      console.error('Failed to update user data:', error);
    }
  };

  return (
      <AuthContext.Provider
          value={{
            user,
            isAuthenticated,
            login,
            logout,
            updateUser,
            loading: isLoading
          }}
      >
        {!isLoading && children}
      </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
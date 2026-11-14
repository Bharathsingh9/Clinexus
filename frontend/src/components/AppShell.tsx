import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import {
  LayoutDashboard,
  PlusCircle,
  Activity,
  FileText,
  Search,
  CheckSquare,
  BarChart2,
  LogOut,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import styles from './AppShell.module.css';

interface AppShellProps {
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({ children }) => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleSignOut = () => {
    logout();
    navigate('/login');
  };

  const navItems = [
    { to: '/', label: 'Dashboard', icon: LayoutDashboard, enabled: true },
    { to: '/submit', label: 'Case submission', icon: PlusCircle, enabled: true },
    { to: '/analysis', label: 'Analysis workflow', icon: Activity, enabled: true },
    { to: '/report', label: 'Explainable report', icon: FileText, enabled: false },
    { to: '/evidence', label: 'Evidence explorer', icon: Search, enabled: false },
    { to: '/review', label: 'Human review', icon: CheckSquare, enabled: false },
    { to: '/evaluations', label: 'Evaluation', icon: BarChart2, enabled: false },
  ];

  return (
    <div className={styles.layout}>
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>

      {/* Permanent Research Disclaimer Banner */}
      <aside
        className={styles.disclaimerBanner}
        role="complementary"
        aria-label="Regulatory Disclaimer"
      >
        Research tool. Not a medical device. Outputs require independent clinical judgment.
      </aside>

      {/* Top Bar */}
      <header className={styles.topBar}>
        <NavLink to="/" className={styles.brand} aria-label="Clinexus Home">
          <span>Clinexus</span>
          <span className={styles.brandBadge}>Research</span>
        </NavLink>

        {user && (
          <div className={styles.userSection}>
            <div className={styles.userInfo}>
              <span className={styles.userName}>{user.email}</span>
              <span className={styles.userRole}>{user.role}</span>
            </div>
            <button
              type="button"
              onClick={handleSignOut}
              className={styles.signOutBtn}
              aria-label="Sign out of Clinexus"
            >
              <LogOut size={14} aria-hidden="true" />
              <span>Sign out</span>
            </button>
          </div>
        )}
      </header>

      {/* Main Container */}
      <div className={styles.mainContainer}>
        {/* Left Sidebar Navigation */}
        <nav className={styles.sidebar} aria-label="Main Navigation">
          {navItems.map((item) => {
            const Icon = item.icon;
            if (!item.enabled) {
              return (
                <div
                  key={item.label}
                  className={`${styles.navItem} ${styles.disabled}`}
                  aria-disabled="true"
                >
                  <div className={styles.navLabelGroup}>
                    <Icon size={16} aria-hidden="true" />
                    <span>{item.label}</span>
                  </div>
                  <span className={styles.comingSoonBadge}>Coming soon</span>
                </div>
              );
            }

            return (
              <NavLink
                key={item.label}
                to={item.to}
                className={({ isActive }) =>
                  `${styles.navItem} ${isActive ? styles.navItemActive : ''}`
                }
              >
                <div className={styles.navLabelGroup}>
                  <Icon size={16} aria-hidden="true" />
                  <span>{item.label}</span>
                </div>
              </NavLink>
            );
          })}
        </nav>

        {/* Main Content Area */}
        <main id="main-content" className={styles.content} tabIndex={-1}>
          {children}
        </main>
      </div>
    </div>
  );
};

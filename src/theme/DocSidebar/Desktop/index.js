import React, { useEffect, useState } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import DocSidebarDesktop from '@theme-original/DocSidebar/Desktop';

function PageViewCounter() {
  const { siteConfig } = useDocusaurusContext();
  const counterUrl = siteConfig.customFields.pageViewCounterUrl;
  const [pageViews, setPageViews] = useState();

  useEffect(() => {
    if (!counterUrl) return undefined;

    const controller = new AbortController();
    const endpoint = `${counterUrl.replace(/\/$/, '')}/view`;

    fetch(endpoint, { method: 'POST', signal: controller.signal })
      .then((response) => (response.ok ? response.json() : Promise.reject(response)))
      .then(({ pageViews: count }) => setPageViews(count))
      .catch(() => {});

    return () => controller.abort();
  }, [counterUrl]);

  if (!counterUrl) return null;

  return (
    <section className="page-view-counter" aria-live="polite">
      Page views: <strong>{pageViews === undefined ? '—' : pageViews.toLocaleString()}</strong>
    </section>
  );
}

export default function DocSidebarDesktopWrapper(props) {
  return (
    <div className="page-view-sidebar-wrapper">
      <DocSidebarDesktop {...props} />
      <PageViewCounter />
    </div>
  );
}

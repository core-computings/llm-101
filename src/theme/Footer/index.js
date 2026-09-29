import React, { useEffect, useRef, useState } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { useLocation } from '@docusaurus/router';
import Footer from '@theme-original/Footer';

function PageViewCounter() {
  const { siteConfig } = useDocusaurusContext();
  const { pathname } = useLocation();
  const counterUrl = siteConfig.customFields.pageViewCounterUrl;
  const lastTrackedPath = useRef();
  const [pageViews, setPageViews] = useState();

  useEffect(() => {
    if (!counterUrl || lastTrackedPath.current === pathname) return undefined;

    lastTrackedPath.current = pathname;
    const controller = new AbortController();
    const endpoint = `${counterUrl.replace(/\/$/, '')}/view`;

    fetch(endpoint, { method: 'POST', signal: controller.signal })
      .then((response) => (response.ok ? response.json() : Promise.reject(response)))
      .then(({ pageViews: count }) => setPageViews(count))
      .catch(() => {});

    return () => controller.abort();
  }, [counterUrl, pathname]);

  if (!counterUrl) return null;

  return (
    <section className="page-view-counter" aria-live="polite">
      Page views: <strong>{pageViews === undefined ? '—' : pageViews.toLocaleString()}</strong>
    </section>
  );
}

export default function FooterWrapper(props) {
  return (
    <>
      <Footer {...props} />
      <PageViewCounter />
    </>
  );
}

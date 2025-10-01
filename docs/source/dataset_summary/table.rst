.. title:: EEG Datasets Table

.. rubric:: EEG Datasets Table

The data in EEG-DaSh originates from a collaboration involving 25 laboratories, encompassing 27,053 participants. This extensive collection includes M-EEG data, which is a combination of EEG and MEG signals. The data is sourced from various studies conducted by these labs,
involving both healthy subjects and clinical populations with conditions such as ADHD, depression, schizophrenia, dementia, autism, and psychosis. Additionally, data spans different mental states like sleep, meditation, and cognitive tasks.

In addition, EEG-DaSh will incorporate a subset of the data converted from `NEMAR <https://nemar.org/>`__, which includes 330 MEEG BIDS-formatted datasets, further expanding the archive with well-curated, standardized neuroelectromagnetic data.

.. raw:: html

   <figure class="eegdash-figure" style="margin: 0 0 1.25rem 0;">

.. raw:: html
   :file: ../_static/dataset_generated/dataset_summary_table.html

.. raw:: html

   <figcaption class="eegdash-caption">
     Table: Sortable catalogue of EEG‑DaSh datasets. Use the “Filters” button to open column filters;
     click a column header to jump directly to a filter pane. The Total row is pinned at the bottom.
     * means that we use the median value across multiple recordings in the dataset, and empty cells
     when the metainformation is not extracted yet.
   </figcaption>
   </figure>

Pathology, modality, and dataset type now surface as consistent color-coded tags so you can scan the table at a glance and reuse the same visual language as the model catalog.

.. raw:: html

  <!-- jQuery + DataTables core -->
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <link rel="stylesheet" href="https://cdn.datatables.net/v/bm/dt-1.13.4/datatables.min.css"/>
  <script src="https://cdn.datatables.net/v/bm/dt-1.13.4/datatables.min.js"></script>

  <!-- Buttons + SearchPanes (+ Select required by SearchPanes) -->
  <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">
  <script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
  <link rel="stylesheet" href="https://cdn.datatables.net/select/1.7.0/css/select.dataTables.min.css">
  <link rel="stylesheet" href="https://cdn.datatables.net/searchpanes/2.3.1/css/searchPanes.dataTables.min.css">
  <script src="https://cdn.datatables.net/select/1.7.0/js/dataTables.select.min.js"></script>
  <script src="https://cdn.datatables.net/searchpanes/2.3.1/js/dataTables.searchPanes.min.js"></script>

  <style>
    /* Styling for the Total row (placed in tfoot) */
    table.sd-table tfoot td {
      font-weight: 600;
      border-top: 2px solid rgba(0,0,0,0.2);
      background: #f9fafb;
      /* Match body cell padding to keep perfect alignment */
      padding: 8px 10px !important;
      vertical-align: middle;
    }

    /* Right-align numeric-like columns (2..8) consistently for body & footer */
    table.sd-table tbody td:nth-child(n+2),
    table.sd-table tfoot td:nth-child(n+2) {
      text-align: right;
    }
    /* Keep first column (Dataset/Total) left-aligned */
    table.sd-table tbody td:first-child,
    table.sd-table tfoot td:first-child {
      text-align: left;
    }
  </style>

  <script>
  // Helper: robustly extract values for SearchPanes when needed
  function tagsArrayFromHtml(html) {
    if (html == null) return [];
    // If it's numeric or plain text, just return as a single value
    if (typeof html === 'number') return [String(html)];
    if (typeof html === 'string' && html.indexOf('<') === -1) return [html.trim()];
    // Else parse any .tag elements inside HTML
    var tmp = document.createElement('div');
    tmp.innerHTML = html;
    var tags = Array.from(tmp.querySelectorAll('.tag')).map(function(el){
      return (el.textContent || '').trim();
    });
    return tags.length ? tags : [tmp.textContent.trim()];
  }

  // Helper: parse human-readable sizes like "4.31 GB" into bytes (number)
  function parseSizeToBytes(text) {
    if (!text) return 0;
    var s = String(text).trim();
    var m = s.match(/([\d,.]+)\s*(TB|GB|MB|KB|B)/i);
    if (!m) return 0;
    var value = parseFloat(m[1].replace(/,/g, ''));
    var unit = m[2].toUpperCase();
    var factor = { B:1, KB:1024, MB:1024**2, GB:1024**3, TB:1024**4 }[unit] || 1;
    return value * factor;
  }

  $(function () {
    var $table = $('#datasets-table');
    if (!$table.length) {
      return;
    }
    if ($.fn.DataTable && $.fn.DataTable.isDataTable($table[0])) {
      return;
    }

    // 1) Move the "Total" row into <tfoot> so sorting/filtering never moves it
    var $tbody = $table.find('tbody');
    var $total = $tbody.find('tr').filter(function(){
      return $(this).find('td').eq(0).text().trim() === 'Total';
    });
    if ($total.length) {
      var $tfoot = $table.find('tfoot');
      if (!$tfoot.length) $tfoot = $('<tfoot/>').appendTo($table);
      $total.appendTo($tfoot);
    }

    // 2) Initialize DataTable with SearchPanes button
    var FILTER_COLS = [1,2,3,4,5,6];
    // Detect the index of the size column by header text
    var sizeIdx = (function(){
      var idx = -1;
      $table.find('thead th').each(function(i){
        var t = $(this).text().trim().toLowerCase();
        if (t === 'size on disk' || t === 'size') idx = i;
      });
      return idx;
    })();

    var table = $table.DataTable({
      dom: 'Blfrtip',
      paging: false,
      searching: true,
      info: false,
      language: {
        search: 'Filter dataset:',
        searchPanes: { collapse: { 0: 'Filters', _: 'Filters (%d)' } }
      },
      buttons: [{
        extend: 'searchPanes',
        text: 'Filters',
        config: { cascadePanes: true, viewTotal: true, layout: 'columns-4', initCollapsed: false }
      }],
      columnDefs: (function(){
        var defs = [
          { searchPanes: { show: true }, targets: FILTER_COLS }
        ];
        if (sizeIdx !== -1) {
          defs.push({
            targets: sizeIdx,
            render: function(data, type) {
              if (type === 'sort' || type === 'type') {
                return parseSizeToBytes(data);
              }
              return data;
            }
          });
        }
        return defs;
      })()
    });

    // 3) UX: click a header to open the relevant filter pane
    $table.find('thead th').each(function (i) {
      if ([1,2,3,4].indexOf(i) === -1) return;
      $(this).css('cursor','pointer').attr('title','Click to filter this column');
      $(this).on('click', function () {
        table.button('.buttons-searchPanes').trigger();
        setTimeout(function () {
          var idx = [1,2,3,4].indexOf(i);
          var $container = $(table.searchPanes.container());
          var $pane = $container.find('.dtsp-pane').eq(idx);
          var $title = $pane.find('.dtsp-title');
          if ($title.length) $title.trigger('click');
        }, 0);
      });
    });
  });
  </script>

// Inject API members into the TOC in the sidebar.
// This should be loaded in the localtoc.html template.

$(function (){
    $('div.section').each(function(index) {
        var $section = $(this),
            $tocitem = $('.sphinxlocaltoc li:has(> a.reference.internal[href="#' + 
                         $section.attr('id') +
                         '"])');
            $members = $('<ul>').appendTo($tocitem);
        $('> dl > dt', $section).each(function(index) {
            var $member = $(this),
                $tocMember = $('<li class="api-member">');
            $tocMember.text($('.property', $member).text() + ' ');
            $tocMember.append('<a href="#' + $member.attr('id') + '">' + 
                              $('.descname', $member).text() + 
                              '</a>');
            $members.append($tocMember);
        });
    });
});
